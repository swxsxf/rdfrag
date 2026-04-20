"""Unified launcher for the RDFRAG API and Gradio chat UI."""

from __future__ import annotations

import argparse
import os

import uvicorn

from rdfrag_vkr.api.app import app as fastapi_app


def _ensure_localhost_no_proxy() -> None:
    hosts = {"127.0.0.1", "localhost", "0.0.0.0"}
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        values = {item.strip() for item in current.split(",") if item.strip()}
        values.update(hosts)
        os.environ[key] = ",".join(sorted(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RDFRAG services.")
    parser.add_argument(
        "--mode",
        choices=("ui", "api"),
        default="ui",
        help="Launch the Gradio UI or the FastAPI API.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind.")
    args = parser.parse_args()

    if args.mode == "api":
        uvicorn.run(fastapi_app, host=args.host, port=args.port)
        return

    from rdfrag_vkr.ui.gradio_app import CUSTOM_CSS, HEAD_HTML, create_demo

    _ensure_localhost_no_proxy()
    ui_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    demo = create_demo()
    demo.launch(
        server_name=ui_host,
        server_port=args.port,
        css=CUSTOM_CSS,
        head=HEAD_HTML,
    )


if __name__ == "__main__":
    main()
