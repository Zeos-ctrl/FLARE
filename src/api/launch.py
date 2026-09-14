"""Console-script entry point to launch the dashboard API server."""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Launch the FLARE dashboard API")
    parser.add_argument("--host", default="127.0.0.1")
    # Deliberately not 8000/3000/5000 — those are common defaults for other
    # local dev tools and easy to collide with (see README troubleshooting note).
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
