"""
Start Streamlit dashboard + ngrok tunnel for public sharing.

Usage:
    python scripts/start_public.py
    python scripts/start_public.py --token YOUR_NGROK_TOKEN   # for persistent URL

Free ngrok: sign up at ngrok.com → copy your authtoken → run with --token once.
Without a token it works but URL changes every restart.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None, help="ngrok authtoken (optional, for stable URL)")
    parser.add_argument("--port",  default=8501, type=int, help="Streamlit port (default 8501)")
    args = parser.parse_args()

    from pyngrok import ngrok, conf

    if args.token:
        conf.get_default().auth_token = args.token
        print(f"  ngrok authtoken set.")

    # Start Streamlit in background
    print(f"\n  Starting Streamlit on port {args.port}...")
    streamlit_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py",
         "--server.port", str(args.port),
         "--server.headless", "true",
         "--server.enableCORS", "false"],
        cwd=ROOT,
    )

    # Give Streamlit a moment to boot
    time.sleep(4)

    # Open ngrok tunnel
    print("  Opening ngrok tunnel...")
    tunnel = ngrok.connect(args.port)
    public_url = tunnel.public_url

    print(f"""
╔══════════════════════════════════════════════════════╗
║         AlgoTrade-X — LIVE PUBLIC URL                ║
║                                                      ║
║  {public_url:<52}║
║                                                      ║
║  Share this link with anyone.                        ║
║  Press Ctrl+C to stop.                               ║
╚══════════════════════════════════════════════════════╝
""")

    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        ngrok.kill()
        streamlit_proc.terminate()


if __name__ == "__main__":
    main()
