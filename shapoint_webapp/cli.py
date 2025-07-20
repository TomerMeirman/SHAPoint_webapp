"""
Command-line interface for SHAPoint WebApp.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def main():
    """Main CLI entry point for shapoint-webapp command."""
    parser = argparse.ArgumentParser(
        description="SHAPoint WebApp - Interactive risk assessment",
        prog="shapoint-webapp"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--port",
        type=int, 
        default=8501,
        help="Port to run the webapp on (default: 8501)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost", 
        help="Host to run the webapp on (default: localhost)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Find the app.py file
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)
    
    # Prepare streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    
    # Set config file as environment variable if specified
    if args.config != "config.yaml":
        import os
        os.environ["SHAPOINT_CONFIG"] = args.config
    
    print(f"🚀 Starting SHAPoint WebApp on http://{args.host}:{args.port}")
    print(f"📋 Using config: {args.config}")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Launch streamlit
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 SHAPoint WebApp stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting webapp: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main() 