"""
Copilot.exe launcher - starts the CopilotAI web server.
Finds the venv Python, kills any existing server on port 8000,
launches uvicorn, and opens the browser.
"""

import os
import sys
import subprocess
import webbrowser
import threading
import time
import signal


def get_app_dir():
    """Get the directory where the exe (or script) lives."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def kill_existing_server(port=8000):
    """Kill any process currently listening on the given port."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, creationflags=0x08000000
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit() and int(pid) != 0:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True, creationflags=0x08000000
                    )
                    print(f"Killed old server process (PID {pid})")
    except Exception as e:
        print(f"Warning: Could not check for existing server: {e}")


def open_browser_delayed(url, delay=2.0):
    """Open the browser after a short delay to let the server start."""
    time.sleep(delay)
    webbrowser.open(url)


def main():
    app_dir = get_app_dir()
    os.chdir(app_dir)

    print("=" * 50)
    print("  CopilotAI Server Launcher")
    print("=" * 50)

    # Find venv python
    venv_python = os.path.join(app_dir, "venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        print(f"\nERROR: Virtual environment not found!")
        print(f"Expected: {venv_python}")
        print(f"\nMake sure the venv folder is next to Copilot.exe")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Kill any existing server on port 8000
    kill_existing_server(8000)

    # Open browser after a short delay
    threading.Thread(
        target=open_browser_delayed,
        args=("http://localhost:8000/control", 2.5),
        daemon=True,
    ).start()

    print(f"\nStarting server at http://localhost:8000")
    print("Press Ctrl+C to stop.\n")

    # Launch uvicorn via the venv python
    try:
        proc = subprocess.run(
            [venv_python, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=app_dir,
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
