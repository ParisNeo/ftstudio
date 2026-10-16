"""Own benchmarks. JSONL, one item per line:
  {"id": "q1", "prompt": "...", "expected": "42", "match": "numeric", "tag": "math"}
  or "messages": [{"role":"system",...},{"role":"user",...}] instead of "prompt".
match: exact (normalized) | contains | regex | numeric (last number in output) | python. Default: exact.
python: the first code block of the answer is RUN (fresh temp dir, 20 s timeout) and must exit 0 with every string of
  "expected" (str or list) in stdout+stderr (ANSI codes stripped). Optional "setup" code, "timeout". This catches invented APIs.
  WARNING: it executes model-written code on this machine, unsandboxed. Use it on your own machine with your own prompts.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def load(path: str) -> list[dict]:
    items = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            it = json.loads(line)
            it.setdefault("id", f"item{i}")
            items.append(it)
    return items


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip(" .!?\"'")


ANSI = re.compile(r"\x1b\[[0-9;]*m")
FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def run_python(code: str, timeout: int = 20):
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-c", code], cwd=d, capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        except subprocess.TimeoutExpired:
            return 124, "", f"timed out after {timeout}s"
    return r.returncode, ANSI.sub("", r.stdout), ANSI.sub("", r.stderr)


def score_detail(item: dict, output: str):
    """(ok, detail). detail explains python-mode failures."""
    if item.get("match") != "python":
        return score(item, output), ""
    blocks = FENCE.findall(output)
    code = (item.get("setup", "") + "\n" if item.get("setup") else "") + (blocks[0] if blocks else output)
    rc, out, err = run_python(code, int(item.get("timeout", 20)))
    if rc != 0:
        return False, f"exit {rc}: {err.strip()[-300:]}"
    exp = item.get("expected") or []
    exp = [exp] if isinstance(exp, str) else exp
    missing = [e for e in exp if e not in out + "\n" + err]
    return not missing, (f"missing in output: {missing}" if missing else "")


def score(item: dict, output: str) -> bool:
    if item.get("match") == "python":
        return score_detail(item, output)[0]
    exp, mode = str(item["expected"]), item.get("match", "exact")
    if mode == "exact":
        return _norm(output) == _norm(exp)
    if mode == "contains":
        return _norm(exp) in _norm(output)
    if mode == "regex":
        return re.search(exp, output, re.S | re.I) is not None
    if mode == "numeric":
        nums = NUM_RE.findall(output)
        try:
            return bool(nums) and abs(float(nums[-1].replace(",", "")) - float(exp.replace(",", ""))) < 1e-6
        except ValueError:
            return False
    raise ValueError(f"unknown match mode {mode!r}")


def run(base_url: str, items: list[dict], out_path: Path, max_tokens: int = 512, model: str = "local") -> dict:
    hits, by_tag = 0, {}
    with httpx.Client(timeout=300) as client, open(out_path, "w", encoding="utf-8") as out:
        for it in items:
            messages = it.get("messages") or [{"role": "user", "content": it["prompt"]}]
            r = client.post(
                f"{base_url}/v1/chat/completions",
                json={"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens},
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            ok, detail = score_detail(it, text)
            hits += ok
            t = by_tag.setdefault(it.get("tag", "all"), [0, 0])
            t[0] += ok
            t[1] += 1
            out.write(json.dumps({"id": it["id"], "ok": ok, "detail": detail, "output": text, "expected": it.get("expected")}) + "\n")
    n = len(items)
    return {
        "n": n,
        "accuracy": hits / n if n else None,
        "by_tag": {k: v[0] / v[1] for k, v in by_tag.items()},
        "records": str(out_path),
    }
