#!/usr/bin/env python3
"""
start_app.py – Launcher for the SHAPoint Streamlit web-app.

Usage
-----
python start_app.py [--port 8501] [--clear-cache] [-- <extra streamlit args>]

Options
-------
--port         Port to serve on (default 8501).
--clear-cache  Delete saved model file and Streamlit cache so the app starts fresh.
--             Separator; anything after it is passed to `streamlit run` verbatim.

The script should be executed from the project root where *app.py* lives.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
MODEL_FILE = ROOT_DIR / "shapoint_webapp" / "models" / "shapoint_cardiovascular_model.pkl"
STREAMLIT_CACHE_DIRS = [ROOT_DIR / "__pycache__", ROOT_DIR / ".streamlit"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the SHAPoint web application")
    parser.add_argument("--port", type=int, default=8501, help="Port to expose the Streamlit server on (default 8501)")
    parser.add_argument("--clear-cache", action="store_true", help="Remove persisted model & Streamlit cache first")
    parser.add_argument("--", dest="streamlit_args", nargs=argparse.REMAINDER, help="Additional arguments forwarded to Streamlit")
    return parser.parse_args()


def _clear_cache() -> None:
    # Remove model pickle so web app retrains automatically
    if MODEL_FILE.exists():
        MODEL_FILE.unlink()
        print(f"🗑️  Deleted existing model 👉 {MODEL_FILE}")
    else:
        print("ℹ️  No existing model file to delete")

    # Attempt to remove known Streamlit cache dirs (non-fatal on failure)
    for cache_dir in STREAMLIT_CACHE_DIRS:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                print(f"🗑️  Cleared cache directory 👉 {cache_dir}")
            except Exception as exc:
                print(f"⚠️  Could not remove {cache_dir}: {exc}")


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.clear_cache:
        _clear_cache()

    launch_url = f"http://localhost:{args.port}"
    print("🚀  Launching SHAPoint WebApp …")
    print(f"🌐  Access it at {launch_url}\n")

    # Build command for `streamlit run app.py`
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT_DIR / "shapoint_webapp" / "app.py"), "--server.port", str(args.port)]
    if args.streamlit_args:
        cmd.extend(args.streamlit_args)

    # Start subprocess; inherit stdout/stderr so logs show immediately
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"❌  Streamlit exited with non-zero status: {exc.returncode}")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main() 