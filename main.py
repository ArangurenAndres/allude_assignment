# main.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_web(host: str, port: int, debug: bool) -> int:
    """
    Run the Flask web app (src.webapp).
    """
    env = dict(**__import__("os").environ)
    env["FLASK_RUN_HOST"] = host
    env["FLASK_RUN_PORT"] = str(port)

    # Run as module so imports work correctly
    cmd = [sys.executable, "-m", "src.webapp"]

    # Debug flag is controlled inside src/webapp.py (app.run(debug=True)).
    # If you want to control it here, change src/webapp.py to read an env var.
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def run_tests() -> int:
    """
    Run the test suite script (run_tests.py).
    """
    script = ROOT / "run_tests.py"
    if not script.exists():
        print("❌ run_tests.py not found in project root.")
        return 1
    return subprocess.call([sys.executable, str(script)], cwd=str(ROOT))


def run_cli() -> int:
    """
    Run your original CLI app (src.app).
    """
    return subprocess.call([sys.executable, "-m", "src.app"], cwd=str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Project entrypoint: web UI, tests, or CLI.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_web = sub.add_parser("web", help="Run the Flask web app")
    p_web.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    p_web.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    p_web.add_argument("--debug", action="store_true", help="Enable debug (optional)")

    sub.add_parser("test", help="Run test_questions.json suite (run_tests.py)")
    sub.add_parser("cli", help="Run the terminal CLI assistant (src.app)")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "web":
        return run_web(args.host, args.port, args.debug)

    if args.command == "test":
        return run_tests()

    if args.command == "cli":
        return run_cli()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
