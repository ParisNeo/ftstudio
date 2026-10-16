import json
import re

import pytest

from ftstudio import config, gguf_catalog
from ftstudio.workers import fetch


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_HOME", str(tmp_path))
    monkeypatch.delenv("FTS_MODELS_DIR", raising=False)


def test_examples_are_well_formed():
    for e in gguf_catalog.BUILTIN:
        assert fetch.REPO.match(e["repo"]) and fetch.FILE.match(e["file"]) and e["name"] and e["note"]
    assert len({(e["repo"], e["file"]) for e in gguf_catalog.BUILTIN}) == len(gguf_catalog.BUILTIN)


def test_user_examples_extend_and_bad_file_is_ignored(tmp_path):
    f = tmp_path / "gguf_examples.json"
    f.write_text(json.dumps([{"repo": "me/m-GGUF", "file": "m-Q4_K_M.gguf"}]))
    assert any(e["repo"] == "me/m-GGUF" and e["name"] == "m-Q4_K_M.gguf" for e in gguf_catalog.examples())
    f.write_text("{oops")
    assert len(gguf_catalog.examples()) == len(gguf_catalog.BUILTIN)


def test_local_scan_finds_downloaded_exported_and_extra(tmp_path, monkeypatch):
    (gguf_catalog.models_dir() / "a__b").mkdir()
    (gguf_catalog.models_dir() / "a__b" / "x-Q4_K_M.gguf").write_bytes(b"0" * 2_000_000)
    (config.runs_dir() / "job1" / "gguf").mkdir(parents=True)
    (config.runs_dir() / "job1" / "gguf" / "tuned.Q4_K_M.gguf").write_bytes(b"0" * 1_000_000)
    extra = tmp_path / "elsewhere"
    extra.mkdir()
    (extra / "mine.gguf").write_bytes(b"0")
    monkeypatch.setenv("FTS_MODELS_DIR", str(extra))
    got = {m["name"]: m["where"] for m in gguf_catalog.local_ggufs()}
    assert got == {"x-Q4_K_M.gguf": "downloaded", "tuned.Q4_K_M.gguf": "exported", "mine.gguf": "other"}


def test_fetch_worker_validates_downloads_and_wraps_errors(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()

    def write(params):
        (run / "params.json").write_text(json.dumps(params))

    for bad in ({"repo": "../x/y", "file": "a.gguf"}, {"repo": "a/b", "file": "../a.gguf"}, {"repo": "a/b", "file": "a.bin"}, {}):
        write(bad)
        with pytest.raises(ValueError):
            fetch.main(str(run))

    def fake(repo, file, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / file).write_bytes(b"0" * 3_000_000)
        return str(dest / file)
    monkeypatch.setattr(fetch, "download", fake)
    write({"repo": "unsloth/Qwen3.5-2B-GGUF", "file": "Qwen3.5-2B-Q4_K_M.gguf"})
    fetch.main(str(run))
    res = json.loads((run / "result.json").read_text())
    assert res["size_mb"] == 3 and res["path"].endswith("Qwen3.5-2B-Q4_K_M.gguf") and "unsloth__Qwen3.5-2B-GGUF" in res["path"]
    assert [m["name"] for m in gguf_catalog.local_ggufs()] == ["Qwen3.5-2B-Q4_K_M.gguf"]      # now offered in the form

    def boom(*a):
        raise OSError("404 Entry Not Found")
    monkeypatch.setattr(fetch, "download", boom)
    with pytest.raises(RuntimeError, match="could not download unsloth/Qwen3.5-2B-GGUF"):
        fetch.main(str(run))


def test_gguf_endpoint_and_fetch_kind():
    from fastapi.testclient import TestClient
    from ftstudio.api import app
    with TestClient(app) as c:
        r = c.get("/gguf").json()
        assert r["local"] == [] and len(r["examples"]) >= 4 and r["models_dir"].endswith("models")
        assert c.post("/jobs", json={"kind": "fetch", "params": {"repo": "a/b", "file": "a.gguf"}}).status_code == 200
