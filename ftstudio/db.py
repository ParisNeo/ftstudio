"""SQLite job table. One row per job; status machine:
queued -> running -> succeeded | failed | cancelled   (running -> cancelling -> cancelled)
"""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from . import config

TERMINAL = {"succeeded", "failed", "cancelled"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, params TEXT NOT NULL,
  status TEXT NOT NULL, created REAL NOT NULL, started REAL, finished REAL,
  result TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created);
"""


@contextmanager
def conn():
    c = sqlite3.connect(config.db_path(), timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    try:
        yield c
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)


def _row(r):
    if r is None:
        return None
    d = dict(r)
    d["params"] = json.loads(d["params"])
    d["result"] = json.loads(d["result"]) if d["result"] else None
    return d


def create_job(kind: str, params: dict) -> str:
    jid = uuid.uuid4().hex[:12]
    with conn() as c:
        c.execute(
            "INSERT INTO jobs(id,kind,params,status,created) VALUES(?,?,?,?,?)",
            (jid, kind, json.dumps(params), "queued", time.time()),
        )
    return jid


def get_job(jid: str):
    with conn() as c:
        return _row(c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone())


def list_jobs(limit: int = 50):
    with conn() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY created DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def claim_next():
    """Atomically move the oldest queued job to running. The single caller IS the GPU lock."""
    with conn() as c:
        c.execute("BEGIN IMMEDIATE")
        r = c.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created, rowid LIMIT 1").fetchone()
        if r is None:
            c.execute("COMMIT")
            return None
        c.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (time.time(), r["id"]))
        c.execute("COMMIT")
    return get_job(r["id"])


def finish(jid: str, status: str, result=None, error=None):
    with conn() as c:
        c.execute(
            "UPDATE jobs SET status=?, finished=?, result=?, error=? WHERE id=?",
            (status, time.time(), json.dumps(result) if result is not None else None, error, jid),
        )


def request_cancel(jid: str):
    """Returns the new status, or None if the job doesn't exist."""
    with conn() as c:
        c.execute("BEGIN IMMEDIATE")
        r = c.execute("SELECT status FROM jobs WHERE id=?", (jid,)).fetchone()
        if r is None:
            c.execute("COMMIT")
            return None
        new = r["status"]
        if new == "queued":
            new = "cancelled"
            c.execute("UPDATE jobs SET status=?, finished=? WHERE id=?", (new, time.time(), jid))
        elif new == "running":
            new = "cancelling"
            c.execute("UPDATE jobs SET status=? WHERE id=?", (new, jid))
        c.execute("COMMIT")
        return new


def recover():
    """Jobs left running by a crash can't be resumed: mark them failed."""
    with conn() as c:
        c.execute(
            "UPDATE jobs SET status='failed', error='interrupted by restart', finished=? "
            "WHERE status IN ('running','cancelling')",
            (time.time(),),
        )
