"""Cross-platform launcher.

Run with ``python app.py`` on Windows, macOS or Linux. It prepares the
application home, initialises the database, and starts Streamlit with the
correct interpreter, so no shell-specific activation step is required.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.constants import APP_NAME, APP_VERSION, ensure_directories  # noqa: E402
from memory.database import init_database  # noqa: E402
from observability.logger import configure_logging  # noqa: E402


def prepare() -> None:
    """Create the application directories and database schema."""
    ensure_directories()
    configure_logging()
    asyncio.run(init_database())


def main() -> int:
    """Prepare the environment and launch the interface."""
    prepare()
    print(f"{APP_NAME} {APP_VERSION}")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(SRC / "ui" / "streamlit_app.py"),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    return subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
