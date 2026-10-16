"""Adapter -> merged 16-bit HF model -> GGUF -> quantized GGUF, using llama.cpp's own tools. For the peft and mlx engines
(Unsloth does this itself). Needs llama.cpp: FTS_LLAMA_CPP = a llama.cpp source folder (has convert_hf_to_gguf.py; its
requirements: pip install -r requirements/requirements-convert_hf_to_gguf.txt) and llama-quantize on PATH or FTS_LLAMA_QUANTIZE."""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_converter() -> Path:
    cands = [Path(os.environ["FTS_LLAMA_CPP"]) / "convert_hf_to_gguf.py"] if os.environ.get("FTS_LLAMA_CPP") else []
    cands += [Path(p) / "convert_hf_to_gguf.py" for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError("convert_hf_to_gguf.py not found. Clone https://github.com/ggml-org/llama.cpp and set FTS_LLAMA_CPP to that folder.")


def find_quantizer() -> str:
    if os.environ.get("FTS_LLAMA_QUANTIZE"):
        return os.environ["FTS_LLAMA_QUANTIZE"]
    root = os.environ.get("FTS_LLAMA_CPP")
    for c in ([Path(root) / "build" / "bin" / n for n in ("llama-quantize", "llama-quantize.exe")] if root else []):
        if c.is_file():
            return str(c)
    exe = shutil.which("llama-quantize")
    if exe:
        return exe
    raise FileNotFoundError("llama-quantize not found. Install a llama.cpp release and put it on PATH, or set FTS_LLAMA_QUANTIZE.")


def merge_peft(base_model: str, adapter: Path, out: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16) if int(__import__("transformers").__version__.split(".")[0]) >= 5 \
        else AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(out))


def merge_mlx(base_model: str, adapter: Path, out: Path):
    subprocess.run([sys.executable, "-m", "mlx_lm.fuse", "--model", base_model, "--adapter-path", str(adapter), "--save-path", str(out)], check=True)


def export_generic(run: Path, p: dict, cfg: dict, adapter: Path, quants: list[str], backend: str, merge=None, run_cmd=None, name=None):
    base = cfg.get("base_model") or p.get("base_model")
    if not base:
        raise ValueError("base_model unknown: export from a train job, or pass base_model")
    run_cmd = run_cmd or (lambda cmd: subprocess.run([str(c) for c in cmd], check=True))
    merged, outdir = run / "merged", run / "gguf"
    outdir.mkdir(exist_ok=True)
    print(f"merging adapter into {base} ({backend})", flush=True)
    (merge or (merge_mlx if backend == "mlx" else merge_peft))(base, adapter, merged)
    name = name or adapter.parent.name
    f16 = outdir / f"{name}-f16.gguf"
    conv = find_converter()
    run_cmd([sys.executable, conv, merged, "--outfile", f16, "--outtype", "f16"])
    for q in quants:
        if q.lower() == "f16":
            continue
        run_cmd([find_quantizer(), f16, outdir / f"{name}-{q.upper()}.gguf", q.upper()])
    if not any(q.lower() == "f16" for q in quants):
        f16.unlink(missing_ok=True)       # keep only what was asked for
    return sorted(outdir.glob("*.gguf"))
