"""Merge a trained adapter, convert to GGUF, quantize, then queue an eval per GGUF. params:
  train_job | adapter_dir     quantizations?=["q4_k_m"]   max_seq_len?=2048   backend? (default: the one the train job used)
  auto_eval?: eval suites, e.g. {"custom": {"file": ...}, "perplexity": {"file": ...}}   name?
Unsloth-trained adapters are exported by Unsloth; peft/mlx adapters via llama.cpp's convert_hf_to_gguf.py + llama-quantize."""
import json
import re
import sys
from pathlib import Path

from ..doctor import require_accelerator
from ..models import quant_warning
from ..train_utils import queue_evals, resolve_adapter


def quant_of(path: Path, wanted: list[str]) -> str:
    low = path.name.lower()
    return next((q for q in wanted if q.lower() in low), re.sub(r"\.gguf$", "", path.name))


def unsloth_export(run: Path, p: dict, cfg: dict, adapter: Path, quants: list[str]):
    require_accelerator()
    from unsloth import FastLanguageModel, FastModel

    L = FastModel if p.get("loader", cfg.get("loader")) == "model" else FastLanguageModel
    model, tok = L.from_pretrained(model_name=str(adapter), max_seq_length=p.get("max_seq_len", 2048),
                                   load_in_4bit=p.get("load_in_4bit", cfg.get("load_in_4bit", True)))
    for q in quants:  # one call per quant: slower (re-merges) but doesn't depend on list support
        print(f"exporting {q}", flush=True)
        model.save_pretrained_gguf(str(run / "gguf"), tok, quantization_method=q)


def main(run_dir: str):
    run = Path(run_dir)
    p = json.loads((run / "params.json").read_text())
    adapter, quants = resolve_adapter(p), p.get("quantizations") or ["q4_k_m"]
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    cfg, backend = {}, p.get("backend")
    if p.get("train_job"):
        res = json.loads((adapter.parent / "result.json").read_text())
        cfg, backend = res.get("config", {}), backend or res.get("backend")
    backend = backend or "unsloth"
    if cfg.get("base_model"):
        w = quant_warning(cfg["base_model"], quants)
        if w:
            print(w, flush=True)
    if backend == "unsloth":
        unsloth_export(run, p, cfg, adapter, quants)
    else:
        from .export_generic import export_generic
        export_generic(run, p, cfg, adapter, quants, backend)
    files = sorted(run.rglob("*.gguf"))
    if not files:
        raise RuntimeError("no .gguf was produced under the run folder; read the log above")
    ggufs = [{"path": str(f), "quant": quant_of(f, quants), "size_mb": round(f.stat().st_size / 1e6, 1)} for f in files]
    evals = queue_evals(ggufs, p["auto_eval"], p.get("name") or adapter.parent.name) if p.get("auto_eval") else []
    (run / "result.json").write_text(json.dumps({"ggufs": ggufs, "eval_jobs": evals, "backend": backend}, indent=2))
    print(json.dumps({"ggufs": ggufs, "eval_jobs": evals}, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
