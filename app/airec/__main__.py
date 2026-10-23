"""Entry point: python -m airec [--port 0] [--no-auth]."""

from __future__ import annotations

import argparse
import faulthandler
import os
import socket
import sys
from pathlib import Path

from airec.core.config import load_settings
from airec.core.logging import setup_logging
from airec.core.paths import AppPaths
from airec.core.security import Session, new_token, remove_session, write_session
from airec.version import __version__


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="airec", description="AI Recogniser core service")
    parser.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (0 = random)")
    parser.add_argument("--data-dir", type=Path, default=None, help="override data directory")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--no-auth", action="store_true", help="disable API token (development only)")
    parser.add_argument("--version", action="version", version=f"AI Recogniser {__version__}")
    return parser.parse_args(argv)


def _bind(port: int) -> tuple[socket.socket, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    actual = sock.getsockname()[1]
    return sock, int(actual)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.data_dir is not None:
        os.environ["AIREC_DATA_DIR"] = str(args.data_dir)

    paths = AppPaths.resolve()
    paths.ensure()
    log_file = setup_logging(paths.logs, args.log_level)
    faulthandler.enable(file=open(paths.logs / "crash.log", "a", encoding="utf-8"))  # noqa: SIM115

    settings = load_settings(paths.config_file)

    auth_enabled = not args.no_auth and os.environ.get("AIREC_AUTH", "").lower() != "off"
    token = new_token() if auth_enabled else "dev-token"
    try:
        sock, port = _bind(args.port)
    except OSError as exc:
        print(f"cannot bind 127.0.0.1:{args.port}: {exc}", file=sys.stderr)
        return 1

    session = Session.create(port=port, token=token, version=__version__)
    write_session(paths.session_file, session)

    from airec.api.app import create_app  # imported late so --help is fast

    app = create_app(
        settings=settings,
        paths=paths,
        token=token,
        auth_enabled=auth_enabled,
        port=port,
    )

    url = f"http://127.0.0.1:{port}"
    print(f"AI Recogniser {__version__} core: {url} (data: {paths.data})", file=sys.stderr)
    print(f"log: {log_file}", file=sys.stderr)
    if not auth_enabled:
        print("WARNING: API auth is disabled (development mode)", file=sys.stderr)

    import uvicorn

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_config=None,
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    try:
        server.run(sockets=[sock])
    finally:
        remove_session(paths.session_file, pid=os.getpid())
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
