from __future__ import annotations

import http.client
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


VIEWER_PATH = Path("/opt/dionisio/browser/index.html")
PUBLIC_PORT = int(os.getenv("DIONISIO_PROXY_PORT", os.getenv("PORT", "7474")))
NEO4J_HOST = "127.0.0.1"
NEO4J_PORT = int(os.getenv("DIONISIO_NEO4J_HTTP_PORT", "7475"))
MAX_BODY_BYTES = 20 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "DionisioNeo4jProxy/1.0"

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in {"/", "/browser", "/browser/", "/browser/index.html"}:
            if path == "/":
                self.send_response(302)
                self.send_header("Location", "/browser/")
                self.end_headers()
                return
            self._serve_viewer()
            return
        if path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "authorization,content-type")
        self.end_headers()

    def _serve_viewer(self) -> None:
        body = VIEWER_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self.send_error(413, "Request body too large")
            return

        body = self.rfile.read(length) if length else None
        hop_by_hop = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
            "host",
        }
        browser_challenge_headers = {"www-authenticate"}
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in hop_by_hop
        }

        conn = http.client.HTTPConnection(NEO4J_HOST, NEO4J_PORT, timeout=30)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
            response_body = response.read()
        except OSError as exc:
            self.send_error(502, f"Neo4j HTTP API unavailable: {exc}")
            return
        finally:
            conn.close()

        self.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            lower_key = key.lower()
            if lower_key not in hop_by_hop and lower_key not in browser_challenge_headers:
                self.send_header(key, value)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PUBLIC_PORT), Handler)
    print(
        f"Dionisio Neo4j proxy listening on 0.0.0.0:{PUBLIC_PORT}; "
        f"forwarding to {NEO4J_HOST}:{NEO4J_PORT}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
