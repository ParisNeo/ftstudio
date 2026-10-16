"""Generate training entries from raw documents with a GGUF (llama.cpp) or an already running OpenAI-compatible server. params:
  gguf | server_url          sources: [files/folders]      library: name      mode?: append|create (overwrite?: true)
  prompt? (must contain {chunk}; {n} {source} optional)   generator_system?   system_prompt? (system turn stored in entries)
  per_chunk?=3  max_chunks?  dry_run? (writes preview.jsonl, never touches the library; default 3 passages)
  judge?=true  verify_code?=true  repair_rounds?=1  must_import?  setup?  timeout?=20  decontaminate_against?: [eval .jsonl]
  chunk_size?=3000 chunk_overlap?=200 include?=[globs] temperature?=0.3 max_tokens?=1800 ctx?=8192 n_gpu_layers?=99 port?=8091
Writes result.json, rejects.jsonl (read it!), preview.jsonl (dry run)."""
import contextlib
import json
import re
import sys
import time
from pathlib import Path

import httpx

from ..data import synth
from .eval import llama_server


def make_chat(base: str, model: str):
    client = httpx.Client(timeout=900)

    def chat(messages, temperature=0.3, max_tokens=1800):
        for attempt in range(3):
            try:
                r = client.post(f"{base}/v1/chat/completions", json={"model": model, "messages": messages,
                                                                      "temperature": temperature, "max_tokens": max_tokens})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, KeyError, ValueError) as e:
                if attempt == 2:
                    raise RuntimeError(f"LLM request failed: {e}")
                time.sleep(2)

    return chat


def main(run_dir: str):
    run = Path(run_dir)
    p = json.loads((run / "params.json").read_text())
    if not p.get("sources") or not p.get("library"):
        raise ValueError("'sources' and 'library' are required")
    if p.get("server_url"):
        cm, model = contextlib.nullcontext(p["server_url"].rstrip("/")), p["server_url"]
    elif p.get("gguf") and Path(p["gguf"]).is_file():
        cm = llama_server(p["gguf"], p.get("port", 8091), p.get("ctx", 8192), p.get("n_gpu_layers", 99), p.get("server_args", []))
        model = Path(p["gguf"]).stem
    else:
        raise FileNotFoundError("give an existing 'gguf' file or a 'server_url'")
    with cm as base:
        res = synth.run(p, make_chat(base, "local"), run, model=model, log=lambda s: print(s, flush=True))
    (run / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("chunks", "generated", "kept", "rejected", "repaired", "target")}, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
