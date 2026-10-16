"""Single worker loop = single GPU lock. Each job runs as a subprocess in runs/<id>/.
Crashes, CUDA leaks and OOMs die with the subprocess, never with the API."""
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from . import config, db

POLL = 1.0
PKG_PARENT = str(Path(__file__).resolve().parents[1])


def kill_tree(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(10)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


class Runner(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="ftstudio-runner")
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            job = db.claim_next()
            if job is None:
                self.stop_event.wait(POLL)
                continue
            try:
                self._execute(job)
            except Exception as e:  # runner must never die
                db.finish(job["id"], "failed", error=repr(e))

    def _execute(self, job: dict):
        jid = job["id"]
        run_dir = config.runs_dir() / jid
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "params.json").write_text(json.dumps(job["params"], indent=2))
        cmd = [sys.executable, "-u", "-m", f"ftstudio.workers.{job['kind']}", str(run_dir)]
        env = {**os.environ, "PYTHONPATH": PKG_PARENT + os.pathsep + os.environ.get("PYTHONPATH", "")}
        kw = {} if os.name == "nt" else {"start_new_session": True}
        outcome = None
        with open(run_dir / "log.txt", "ab") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, **kw)
            while proc.poll() is None:
                if self.stop_event.is_set():
                    kill_tree(proc)
                    outcome = "interrupted"
                    break
                j = db.get_job(jid)
                if j and j["status"] == "cancelling":
                    kill_tree(proc)
                    outcome = "cancelled"
                    break
                self.stop_event.wait(0.3)
        if outcome == "cancelled":
            db.finish(jid, "cancelled")
        elif outcome == "interrupted":
            db.finish(jid, "failed", error="interrupted by shutdown")
        elif proc.returncode == 0:
            rf = run_dir / "result.json"
            db.finish(jid, "succeeded", result=json.loads(rf.read_text()) if rf.exists() else None)
        else:
            db.finish(jid, "failed", error=f"exit code {proc.returncode}")
