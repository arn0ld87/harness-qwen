#!/usr/bin/env python3
"""A stand-in for ``llama-server`` that starts in milliseconds.

The supervisor's job is process lifecycle, not inference: start it, wait for
health, notice when it dies, stop it without leaving an orphan. Testing that
against the 35B model would mean 25 s per case and a GPU in CI, and would
exercise llama.cpp rather than the supervisor.

Behaviour is driven by flags so one stub covers every case the supervisor has
to classify:

    --port N            listen here (required)
    --host H            bind address
    --ready-after S     answer /health with 503 for this many seconds first
    --die-after S       exit with --exit-code after this long
    --exit-code N       status to die with
    --fail-to-start     exit immediately, before binding
    --leak-secret       print an API key to stdout, to prove redaction
    --model PATH        accepted and echoed, like the real binary
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

_started = time.monotonic()
_ready_after = 0.0
_model = "stub"


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802  (http.server's spelling)
        loading = (time.monotonic() - _started) < _ready_after
        if self.path == "/health":
            if loading:
                self._json(503, {"status": "loading model"})
            else:
                self._json(200, {"status": "ok"})
        elif self.path == "/props":
            self._json(
                200,
                {
                    "model_path": _model,
                    "default_generation_settings": {"n_ctx": 4096},
                    "total_slots": 1,
                },
            )
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": "stub"}]})
        else:
            self._json(404, {"error": "not found"})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default stderr access log; the tests read our stdout."""


def main() -> int:
    global _ready_after, _model
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ready-after", type=float, default=0.0)
    parser.add_argument("--die-after", type=float, default=None)
    parser.add_argument("--exit-code", type=int, default=1)
    parser.add_argument("--fail-to-start", action="store_true")
    parser.add_argument("--leak-secret", action="store_true")
    parser.add_argument("--model", default="stub")
    args, _unknown = parser.parse_known_args()

    _model = args.model
    _ready_after = args.ready_after

    print(f"stub-server starting on {args.host}:{args.port}", flush=True)
    if args.leak_secret:
        print("loading with api_key=sk-live-abcdef1234567890", flush=True)

    if args.fail_to_start:
        print("error: could not load model", file=sys.stderr, flush=True)
        return 2

    server = HTTPServer((args.host, args.port), Handler)
    if args.die_after is not None:
        def kill_later() -> None:
            time.sleep(args.die_after)
            print("stub-server dying on request", flush=True)
            # os._exit: a hard exit, like a segfault or an OOM kill, which is
            # the shape of death the supervisor has to survive.
            import os

            os._exit(args.exit_code)

        threading.Thread(target=kill_later, daemon=True).start()

    print(f"stub-server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
