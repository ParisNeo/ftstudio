"""Named libraries of training entries: $FTS_HOME/library/<name>.jsonl, one {"messages": [...], "meta": {...}} per line.
Prepare-data reads them directly (param "libraries"); the synth worker writes them."""
import json
import re

from . import config

NAME = re.compile(r"^[A-Za-z0-9][\w.-]{0,63}$")


def path(name: str):
    if not NAME.match(name or "") or ".." in name:
        raise ValueError(f"bad library name {name!r}: letters, digits, '.', '_' or '-' (max 64)")
    d = config.home() / "library"
    d.mkdir(exist_ok=True)
    return d / f"{name}.jsonl"


def read_all(name: str) -> list[dict]:
    p = path(name)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def list_libraries() -> list[dict]:
    d = config.home() / "library"
    if not d.exists():
        return []
    return [{"name": f.stem, "entries": sum(1 for l in f.read_text(encoding="utf-8").splitlines() if l.strip()),
             "bytes": f.stat().st_size} for f in sorted(d.glob("*.jsonl"))]


def append(name: str, entries: list[dict]):
    with open(path(name), "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def reset(name: str):
    path(name).write_text("", encoding="utf-8")


def delete(name: str, idx: int):
    rows = read_all(name)
    if not 0 <= idx < len(rows):
        raise IndexError(idx)
    rows.pop(idx)
    path(name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
