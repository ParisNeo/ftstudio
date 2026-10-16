"""Paths and binaries. Read from the environment at call time (keeps tests simple)."""
import os
from pathlib import Path


def home() -> Path:
    p = Path(os.environ.get("FTS_HOME", "./data")).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    return home() / "jobs.sqlite3"


def runs_dir() -> Path:
    p = home() / "runs"
    p.mkdir(exist_ok=True)
    return p


def llama_server_bin() -> str:
    return os.environ.get("FTS_LLAMA_SERVER", "llama-server")


def llama_perplexity_bin() -> str:
    return os.environ.get("FTS_LLAMA_PERPLEXITY", "llama-perplexity")
