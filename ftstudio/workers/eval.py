"""Eval a GGUF through the real llama.cpp binaries (tests the artifact you ship).
params:
  gguf, name?, ctx?=4096, n_gpu_layers?=99, port?=8089, server_args?=[]
  suites: {
    perplexity?: {file, ctx?=512},
    custom?:     {file, max_tokens?=512},
    lm_eval?:    {tasks:[...], limit?, num_fewshot?, tokenizer?}   # see README: verify
  }
"""
import json
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from .. import config
from ..evals import custom

PPL_RE = re.compile(r"Final estimate: PPL = ([0-9.]+)\s*\+/-\s*([0-9.]+)")


def say(*a):
    print(*a, flush=True)


def run_perplexity(gguf: str, cfg: dict, ngl: int) -> dict:
    ctx = cfg.get("ctx", 512)
    cmd = [config.llama_perplexity_bin(), "-m", gguf, "-f", cfg["file"], "-c", str(ctx), "-ngl", str(ngl)]
    say("$", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    last = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        m = PPL_RE.search(line)
        if m:
            last = (float(m[1]), float(m[2]))
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"llama-perplexity exit {rc}")
    if last is None:
        raise RuntimeError("no 'Final estimate: PPL' line in llama-perplexity output (format changed?)")
    return {"ppl": last[0], "stderr": last[1], "ctx": ctx}


@contextmanager
def llama_server(gguf: str, port: int, ctx: int, ngl: int, extra: list):
    cmd = [config.llama_server_bin(), "-m", gguf, "--host", "127.0.0.1", "--port", str(port),
           "-c", str(ctx), "-ngl", str(ngl), *extra]
    say("$", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 600
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early ({proc.returncode})")
            try:
                if httpx.get(base + "/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                raise TimeoutError("llama-server not ready after 600s")
            time.sleep(1)
        say("server ready")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_lm_eval(base: str, cfg: dict, name: str, out_dir: Path) -> dict:
    margs = f"model={name},base_url={base}/v1/completions,num_concurrent=1,max_retries=3"
    if cfg.get("tokenizer"):
        margs += f",tokenizer={cfg['tokenizer']},tokenizer_backend=huggingface"
    cmd = [sys.executable, "-m", "lm_eval", "--model", "local-completions", "--model_args", margs,
           "--tasks", ",".join(cfg["tasks"]), "--output_path", str(out_dir)]
    if cfg.get("limit"):
        cmd += ["--limit", str(cfg["limit"])]
    if "num_fewshot" in cfg:
        cmd += ["--num_fewshot", str(cfg["num_fewshot"])]
    say("$", " ".join(cmd))
    rc = subprocess.call(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"lm_eval exit {rc}")
    files = sorted(out_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError("no lm-eval results_*.json found")
    return json.loads(files[-1].read_text()).get("results", {})


def main(run_dir: str):
    run = Path(run_dir)
    p = json.loads((run / "params.json").read_text())
    gguf = p["gguf"]
    if not Path(gguf).is_file():
        raise FileNotFoundError(gguf)
    name = p.get("name", Path(gguf).stem)
    ngl, ctx, suites = p.get("n_gpu_layers", 99), p.get("ctx", 4096), p.get("suites", {})
    result = {"model": name, "gguf": gguf}

    if "perplexity" in suites:  # first: needs the GPU to itself, no server running
        result["perplexity"] = run_perplexity(gguf, suites["perplexity"], ngl)

    if "custom" in suites or "lm_eval" in suites:
        with llama_server(gguf, p.get("port", 8089), ctx, ngl, p.get("server_args", [])) as base:
            if "custom" in suites:
                c = suites["custom"]
                result["custom"] = custom.run(base, custom.load(c["file"]), run / "custom_records.jsonl",
                                              max_tokens=c.get("max_tokens", 512), model=name)
            if "lm_eval" in suites:
                result["lm_eval"] = run_lm_eval(base, suites["lm_eval"], name, run / "lm_eval")

    (run / "result.json").write_text(json.dumps(result, indent=2))
    say(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
