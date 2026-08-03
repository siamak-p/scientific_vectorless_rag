"""Create the local SQLite database and every application table.

Scientific RAG creates its storage automatically the first time the UI starts,
so running this script is optional. It exists for deployments that want the
database prepared ahead of time, or to verify the storage location.

Usage:
    python init_database.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from core.constants import APP_HOME, CHECKPOINT_FILE, DATABASE_FILE, ensure_directories
from memory.database import init_database


async def _setup() -> None:
    ensure_directories()
    await init_database()


def main() -> int:
    print("Scientific RAG - database initialisation")
    print(f"  application home : {APP_HOME}")
    print(f"  database file    : {DATABASE_FILE}")
    print(f"  checkpoint file  : {CHECKPOINT_FILE}")
    asyncio.run(_setup())
    print("\nAll tables are ready. Start the application with: python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
