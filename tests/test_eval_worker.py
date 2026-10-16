import json
import os
import socket
import stat
import sys
from pathlib import Path

import pytest

from ftstudio.workers import eval as ev

pytestmark = pytest.mark.skipif(os.name == "nt", reason="uses sh wrappers")
FAKE = Path(__file__).with_name("fake_llama.py")


def wrapper(path: Path, mode: str):
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" {mode} "$@"\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_eval_worker_perplexity_and_custom(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_HOME", str(tmp_path))
    monkeypatch.setenv("FTS_LLAMA_SERVER", wrapper(tmp_path / "srv", "server"))
    monkeypatch.setenv("FTS_LLAMA_PERPLEXITY", wrapper(tmp_path / "ppl", "ppl"))
    (tmp_path / "m.gguf").write_bytes(b"x")
    (tmp_path / "wiki.txt").write_text("text")
    (tmp_path / "t.jsonl").write_text(
        json.dumps({"id": "a", "prompt": "2+2?", "expected": "4", "match": "numeric", "tag": "math"}) + "\n"
        + json.dumps({"id": "b", "prompt": "capital?", "expected": "paris", "tag": "geo"}) + "\n"
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "params.json").write_text(json.dumps({
        "gguf": str(tmp_path / "m.gguf"), "port": free_port(),
        "suites": {"perplexity": {"file": str(tmp_path / "wiki.txt")},
                   "custom": {"file": str(tmp_path / "t.jsonl")}},
    }))
    ev.main(str(run))
    r = json.loads((run / "result.json").read_text())
    assert r["perplexity"]["ppl"] == 7.1234
    assert r["custom"]["n"] == 2 and r["custom"]["accuracy"] == 0.5
    assert r["custom"]["by_tag"] == {"math": 1.0, "geo": 0.0}
