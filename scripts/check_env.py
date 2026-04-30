#!/usr/bin/env python3
"""
Quick environment sanity check for the Hermes Self-Healing YouTube RAG Agent repo.
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
