import json
import time

import pytest
from fastapi.testclient import TestClient

from ftstudio import db
from ftstudio.evals import custom


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_HOME", str(tmp_path))
    db.init()


def wait(client, jid, want, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{jid}").json()
        if j["status"] in want:
            return j
        time.sleep(0.2)
    raise AssertionError(f"timeout, last={j['status']}")


def test_claim_is_fifo_and_exclusive():
    a = db.create_job("noop", {})
    b = db.create_job("noop", {})
    assert db.claim_next()["id"] == a
    assert db.claim_next()["id"] == b
    assert db.claim_next() is None


def test_cancel_queued_and_unknown():
    a = db.create_job("noop", {})
    assert db.request_cancel(a) == "cancelled"
    assert db.claim_next() is None
    assert db.request_cancel("nope") is None


def test_api_job_lifecycle_and_stream():
    from ftstudio.api import app

    with TestClient(app) as c:
        assert c.post("/jobs", json={"kind": "bogus"}).status_code == 400
        jid = c.post("/jobs", json={"kind": "noop", "params": {"x": 1}}).json()["id"]
        j = wait(c, jid, {"succeeded", "failed"})
        assert j["status"] == "succeeded" and j["result"] == {"echo": {"x": 1}}
        assert "noop done" in c.get(f"/jobs/{jid}/log").json()["text"]
        assert c.get(f"/jobs/{jid}/artifacts/result.json").json() == {"echo": {"x": 1}}
        assert c.get(f"/jobs/{jid}/artifacts/..%2Fjobs.sqlite3").status_code in (400, 404)
        assert c.get("/jobs/nope/artifacts/log.txt").status_code == 404
        with c.websocket_connect(f"/jobs/{jid}/stream") as ws:
            assert "noop start" in ws.receive_text()


def test_failure_and_cancel_running():
    from ftstudio.api import app

    with TestClient(app) as c:
        bad = c.post("/jobs", json={"kind": "noop", "params": {"fail": True}}).json()["id"]
        assert wait(c, bad, {"failed"})["error"] == "exit code 3"
        slow = c.post("/jobs", json={"kind": "noop", "params": {"sleep": 30}}).json()["id"]
        wait(c, slow, {"running"})
        time.sleep(0.5)
        c.post(f"/jobs/{slow}/cancel")
        assert wait(c, slow, {"cancelled"}, timeout=15)["status"] == "cancelled"


@pytest.mark.parametrize("mode,exp,out,ok", [
    ("exact", "Paris", " paris. ", True),
    ("contains", "paris", "It is Paris, France", True),
    ("regex", r"\bparis\b", "Paris", True),
    ("numeric", "1,234", "so the answer is 1234", True),
    ("numeric", "42", "I think 41", False),
    ("exact", "a", "b", False),
])
def test_scorer(mode, exp, out, ok):
    assert custom.score({"expected": exp, "match": mode}, out) is ok


def test_custom_load(tmp_path):
    f = tmp_path / "t.jsonl"
    f.write_text(json.dumps({"prompt": "hi", "expected": "x"}) + "\n\n")
    assert custom.load(str(f))[0]["id"] == "item0"
