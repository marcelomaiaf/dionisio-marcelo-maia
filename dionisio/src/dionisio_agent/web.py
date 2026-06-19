from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from agents import MaxTurnsExceeded, Runner, SQLiteSession
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel, Field

from dionisio_agent.config import Settings
from dionisio_agent.factory import create_runtime
from dionisio_agent.sessions import create_limited_sqlite_session
from dionisio_agent.tools import ToolRuntime

logger = logging.getLogger(__name__)

AGENT_MAX_TURNS_MESSAGE = (
    "Nao consegui concluir essa operacao em poucas etapas. Para avancar mais rapido, "
    "envie o nome ou ID exato do registro e a acao desejada em uma unica mensagem."
)
AGENT_EMPTY_OUTPUT_MESSAGE = (
    "Nao recebi uma resposta final do modelo neste turno. Tente reenviar a mensagem "
    "ou repetir o pedido com o ID do registro."
)


class WebChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(default="default", min_length=1, max_length=128)


class WebChatResponse(BaseModel):
    ok: bool = True
    session_id: str
    reply: str


class SessionLockRegistry:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def get(self, session_id: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_id] = lock
            return lock


class WebChatBridge:
    def __init__(
        self,
        *,
        agent: Any,
        session_db_path: str,
        timeout_seconds: float = 0,
        max_turns: int,
        session_history_limit: int | None = None,
        runtime: ToolRuntime | None = None,
        locks: SessionLockRegistry | None = None,
    ) -> None:
        self.agent = agent
        self.runtime = runtime
        self.session_db_path = session_db_path
        self.timeout_seconds = timeout_seconds
        self.max_turns = max_turns
        self.session_history_limit = session_history_limit
        self.locks = locks or SessionLockRegistry()

    async def reply(self, *, message: str, session_id: str) -> str:
        prompt = message.strip()
        if not prompt:
            return "Envie uma mensagem para conversar com o agente."

        Path(self.session_db_path).parent.mkdir(parents=True, exist_ok=True)
        normalized_session_id = build_web_chat_session_id(session_id)
        lock = await self.locks.get(normalized_session_id)
        async with lock:
            session = create_limited_sqlite_session(
                normalized_session_id,
                db_path=self.session_db_path,
                history_limit=self.session_history_limit,
            )
            started = time.perf_counter()
            try:
                result = await self._run_agent(prompt=prompt, session=session)
            except MaxTurnsExceeded:
                logger.warning(
                    "web_chat_agent_max_turns_exceeded",
                    extra={
                        "session_id": normalized_session_id,
                        "max_turns": self.max_turns,
                        "latency_seconds": round(time.perf_counter() - started, 4),
                    },
                )
                return AGENT_MAX_TURNS_MESSAGE
            logger.info(
                "web_chat_agent_completed",
                extra={
                    "session_id": normalized_session_id,
                    "latency_seconds": round(time.perf_counter() - started, 4),
                },
            )
        return _result_text(result)

    async def _run_agent(self, *, prompt: str, session: SQLiteSession) -> Any:
        return await Runner.run(
            self.agent,
            prompt,
            session=session,
            max_turns=self.max_turns,
        )

    async def stream_reply(self, *, message: str, session_id: str) -> AsyncIterator[dict[str, Any]]:
        prompt = message.strip()
        if not prompt:
            yield {
                "event": "error",
                "data": {"message": "Envie uma mensagem para conversar com o agente."},
            }
            return

        Path(self.session_db_path).parent.mkdir(parents=True, exist_ok=True)
        normalized_session_id = build_web_chat_session_id(session_id)
        lock = await self.locks.get(normalized_session_id)
        async with lock:
            session = create_limited_sqlite_session(
                normalized_session_id,
                db_path=self.session_db_path,
                history_limit=self.session_history_limit,
            )
            started = time.perf_counter()
            accumulated: list[str] = []
            try:
                result = Runner.run_streamed(
                    self.agent,
                    prompt,
                    session=session,
                    max_turns=self.max_turns,
                )
                async for event in result.stream_events():
                    if event.type != "raw_response_event":
                        continue
                    if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                        accumulated.append(event.data.delta)
                        yield {"event": "delta", "data": {"text": event.data.delta}}
            except MaxTurnsExceeded:
                logger.warning(
                    "web_chat_agent_max_turns_exceeded",
                    extra={
                        "session_id": normalized_session_id,
                        "max_turns": self.max_turns,
                        "latency_seconds": round(time.perf_counter() - started, 4),
                    },
                )
                yield {"event": "delta", "data": {"text": AGENT_MAX_TURNS_MESSAGE}}
                yield {"event": "done", "data": {}}
                return
            final_output = _result_text(result, fallback="".join(accumulated))
            logger.info(
                "web_chat_agent_stream_completed",
                extra={
                    "session_id": normalized_session_id,
                    "latency_seconds": round(time.perf_counter() - started, 4),
                    "streamed_chars": len("".join(accumulated)),
                    "final_chars": len(final_output),
                },
            )
        yield {"event": "final", "data": {"reply": final_output}}
        yield {"event": "done", "data": {}}


def create_app(
    *,
    settings: Settings | None = None,
    web_chat_bridge: WebChatBridge | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime: ToolRuntime | None = None
        warmup_task: asyncio.Task[None] | None = None
        if app.state.web_chat_bridge is None:
            agent, runtime = await create_runtime(settings)
            app.state.runtime = runtime
            if runtime.knowledge_base is not None:
                warmup_task = asyncio.create_task(asyncio.to_thread(runtime.knowledge_base.warmup))
                app.state.graphrag_warmup_task = warmup_task
            app.state.web_chat_bridge = WebChatBridge(
                agent=agent,
                runtime=runtime,
                session_db_path=settings.web_chat_session_db_path,
                max_turns=settings.agent_max_turns,
                session_history_limit=settings.web_chat_session_history_limit,
            )
        try:
            yield
        finally:
            if warmup_task is not None and not warmup_task.done():
                warmup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await warmup_task
            if runtime is not None and runtime.knowledge_base is not None:
                runtime.knowledge_base.close()

    app = FastAPI(title="Dionisio Web Chat", lifespan=lifespan)
    app.state.settings = settings
    app.state.web_chat_bridge = web_chat_bridge
    app.state.web_chat_bridge_lock = asyncio.Lock()
    app.state.runtime = None
    app.state.graphrag_warmup_task = None

    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    if frontend_dir.exists():
        app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

        @app.get("/", response_class=FileResponse)
        async def chat_page() -> FileResponse:
            return FileResponse(frontend_dir / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "dionisio-web-chat",
        }

    @app.post("/api/chat", response_model=WebChatResponse)
    async def chat(request: WebChatRequest) -> WebChatResponse:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required.")
        try:
            web_bridge = await get_web_chat_bridge(app, settings)
            reply = await web_bridge.reply(
                message=message,
                session_id=request.session_id,
            )
        except Exception:
            logger.exception("web_chat_agent_error")
            raise HTTPException(status_code=500, detail="Agent response failed.") from None
        return WebChatResponse(session_id=request.session_id, reply=reply)

    @app.post("/api/chat/stream")
    async def chat_stream(request: WebChatRequest) -> StreamingResponse:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required.")

        async def generate() -> AsyncIterator[str]:
            try:
                web_bridge = await get_web_chat_bridge(app, settings)
                async for item in web_bridge.stream_reply(
                    message=message,
                    session_id=request.session_id,
                ):
                    yield _sse(item["event"], item["data"])
            except Exception:
                logger.exception("web_chat_agent_stream_error")
                yield _sse("error", {"message": "Agent response failed."})
            yield _sse("done", {})

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


async def get_web_chat_bridge(app: FastAPI, settings: Settings) -> WebChatBridge:
    if app.state.web_chat_bridge is None:
        async with app.state.web_chat_bridge_lock:
            if app.state.web_chat_bridge is None:
                agent, runtime = await create_runtime(settings)
                app.state.runtime = runtime
                app.state.web_chat_bridge = WebChatBridge(
                    agent=agent,
                    runtime=runtime,
                    session_db_path=settings.web_chat_session_db_path,
                    max_turns=settings.agent_max_turns,
                    session_history_limit=settings.web_chat_session_history_limit,
                )
    return app.state.web_chat_bridge


def build_web_chat_session_id(session_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", session_id.strip() or "default")
    return f"web:{normalized}"


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _result_text(result: Any, *, fallback: str = "") -> str:
    output = getattr(result, "final_output", None)
    text = str(output) if output is not None else fallback
    if not text.strip():
        return AGENT_EMPTY_OUTPUT_MESSAGE
    return text


app = create_app()


def main() -> None:
    uvicorn.run("dionisio_agent.web:app", host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
