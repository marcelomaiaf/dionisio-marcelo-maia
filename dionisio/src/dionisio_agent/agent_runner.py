from __future__ import annotations

from typing import Any

from agents import Runner

AGENT_MAX_TURNS_MESSAGE = (
    "Nao consegui concluir essa operacao em poucas etapas. Para avancar mais rapido, "
    "envie o nome ou ID exato do registro e a acao desejada em uma unica mensagem."
)
AGENT_EMPTY_OUTPUT_MESSAGE = (
    "Nao recebi uma resposta final do modelo neste turno. Tente reenviar a mensagem "
    "ou repetir o pedido com o ID do registro."
)
FINALIZE_AFTER_TOOL_PROMPT = (
    "Finalize a resposta anterior para o usuario usando o resultado da ferramenta. "
    "Se ja houver dados suficientes, nao chame novas ferramentas. Se faltar aprovacao "
    "humana ou dado obrigatorio, peca isso claramente."
)


async def run_agent_until_text(
    agent: Any,
    prompt: str,
    *,
    session: Any | None = None,
    max_turns: int,
    finalization_attempts: int = 1,
) -> Any:
    result = await _run_agent(agent, prompt, session=session, max_turns=max_turns)
    for _ in range(max(0, finalization_attempts)):
        if _has_final_output_text(result) or not ended_with_tool_output(result):
            return result
        result = await _run_agent(
            agent,
            FINALIZE_AFTER_TOOL_PROMPT,
            session=session,
            max_turns=max_turns,
        )
    return result


def result_text(result: Any, *, fallback: str = "") -> str:
    output = getattr(result, "final_output", None)
    text = str(output) if output is not None else fallback
    if text.strip():
        return text
    message_text = latest_message_output_text(result)
    if message_text.strip():
        return message_text
    return AGENT_EMPTY_OUTPUT_MESSAGE


def ended_with_tool_output(result: Any) -> bool:
    new_items = list(getattr(result, "new_items", []) or [])
    if not new_items:
        return False
    return getattr(new_items[-1], "type", None) == "tool_call_output_item"


def latest_message_output_text(result: Any) -> str:
    for item in reversed(list(getattr(result, "new_items", []) or [])):
        if getattr(item, "type", None) != "message_output_item":
            continue
        raw_item = getattr(item, "raw_item", None)
        content = getattr(raw_item, "content", None)
        if isinstance(raw_item, dict):
            content = raw_item.get("content")
        text = _content_text(content)
        if text.strip():
            return text
    return ""


async def _run_agent(agent: Any, prompt: str, *, session: Any | None, max_turns: int) -> Any:
    kwargs: dict[str, Any] = {"max_turns": max_turns}
    if session is not None:
        kwargs["session"] = session
    return await Runner.run(agent, prompt, **kwargs)


def _has_final_output_text(result: Any) -> bool:
    output = getattr(result, "final_output", None)
    return output is not None and bool(str(output).strip())


def _content_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        text = getattr(part, "text", None)
        if isinstance(part, dict):
            text = part.get("text")
        if text:
            parts.append(str(text))
    return "".join(parts)
