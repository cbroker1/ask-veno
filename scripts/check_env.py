#!/usr/bin/env python3
"""
check_env.py -- Environment sanity check for the Hermes YouTube RAG project.

Runs a series of checks to verify that all required dependencies, tools,
and runtime components are present and working before you start developing
or running the YouTube RAG agent.

USAGE
-----
    python scripts/check_env.py

OPTIONS
    (none)  The script accepts no command-line arguments. All checks are
            run by default.

EXAMPLES
    # Run the check using the system Python
    python scripts/check_env.py

    # Run inside the project's virtual environment
    source .venv/bin/activate   # or .venv\\Scripts\\activate on Windows
    python scripts/check_env.py

    # Run via the shebang (Linux / macOS)
    chmod +x scripts/check_env.py
    ./scripts/check_env.py

WHAT IT CHECKS
--------------
The script performs four categories of checks, in order:

1. Python version
   Verifies that Python 3.11 or newer is installed. The project
   recommends 3.11+ for compatibility with modern dependencies.

2. Required Python modules
   Confirms that each of the following packages is importable:

     pandas        - Data manipulation and analysis
     pydantic      - Data validation using Python type annotations
     python-dotenv - Loading environment variables from .env files
     rich          - Rich text and beautiful formatting in the terminal
     tqdm          - Progress bar display
     chromadb      - Embedding database for vector similarity search
     sentence-transformers - Sentence embeddings for RAG retrieval
     duckdb        - Analytical SQL database

3. Required system commands
   Verifies that these executables are on your PATH:

     ffmpeg    - Audio/video codec and processing pipeline (used for
                 extracting audio from YouTube videos)
     yt-dlp    - YouTube (and other sites) video downloader
     sqlite3   - SQLite command-line shell (used for inspecting the
                 local vector store and caches)

4. Optional local model runtime
   Checks whether ``ollama`` is installed. If found, it also runs
   ``ollama list`` to show which models are currently pulled.
   Missing Ollama is not a failure -- it is informational only.

EXIT CODES
    0  All required checks passed.
    1  One or more required checks failed.

TROUBLESHOOTING
---------------
   Missing Python module
      Install it with pip, e.g.:
          pip install pandas pydantic dotenv rich tqdm chromadb \
                     sentence-transformers duckdb

   Missing system command
      Install via your package manager. Examples:

          # Debian / Ubuntu
          sudo apt install ffmpeg yt-dlp sqlite3

          # macOS (Homebrew)
          brew install ffmpeg yt-dlp sqlite3

          # Arch Linux
          sudo pacman -S ffmpeg yt-dlp sqlite3

   Missing Ollama
      Install from https://ollama.com . After installation, pull a
      model (e.g., ``ollama pull llama3.2``). This is optional but
      required for local inference.

   Still failing?
      Make sure your virtual environment is activated and that the
      correct Python interpreter is on your PATH before running this
      script.

ENVIRONMENT VARIABLES
---------------------
    (none)  This script does not read any environment variables.
            All checks are hard-coded.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys


REQUIRED_PYTHON_MODULES = [
    "pandas",
    "pydantic",
    "dotenv",
    "rich",
    "tqdm",
    "chromadb",
    "sentence_transformers",
    "duckdb",
]

REQUIRED_COMMANDS = [
    "ffmpeg",
    "yt-dlp",
    "sqlite3",
]


def check_python_version() -> bool:
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("  ERROR: Python 3.11+ is recommended.")
        return False
    print("  OK")
    return True


def check_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        print(f"Python module {module_name}: OK")
        return True
    except Exception as exc:
        print(f"Python module {module_name}: ERROR - {exc}")
        return False


def check_command(command: str) -> bool:
    path = shutil.which(command)
    if not path:
        print(f"Command {command}: ERROR - not found on PATH")
        return False

    try:
        result = subprocess.run(
            [command, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        first_line = (result.stdout or result.stderr).splitlines()[0]
        print(f"Command {command}: OK - {first_line}")
        return True
    except Exception as exc:
        print(f"Command {command}: FOUND at {path}, but version check failed - {exc}")
        return False


def check_ollama_optional() -> None:
    path = shutil.which("ollama")
    if not path:
        print("Optional command ollama: not found")
        return

    print(f"Optional command ollama: OK - {path}")
    try:
        result = subprocess.run(
            ["ollama", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("Ollama list:")
            print(result.stdout.strip() or "  No models found.")
        else:
            print("Ollama exists, but `ollama list` failed:")
            print(result.stderr.strip())
    except Exception as exc:
        print(f"Ollama check failed: {exc}")


def main() -> int:
    print("=== Hermes YouTube RAG Environment Check ===")

    ok = True
    ok = check_python_version() and ok

    print("\n=== Python modules ===")
    for module in REQUIRED_PYTHON_MODULES:
        ok = check_module(module) and ok

    print("\n=== Commands ===")
    for command in REQUIRED_COMMANDS:
        ok = check_command(command) and ok

    print("\n=== Optional local model runtime ===")
    check_ollama_optional()

    print("\n=== Result ===")
    if ok:
        print("Environment looks good.")
        return 0

    print("Environment has missing or broken dependencies.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
