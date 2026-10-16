"""GGUF models: example downloads for testing, and the .gguf files already on this computer.
Add your own examples in $FTS_HOME/gguf_examples.json: [{"repo","file","name","size","note"}].
Extra folders to scan: FTS_MODELS_DIR (os.pathsep-separated). Downloads land in $FTS_HOME/models/."""
import json
import os
from pathlib import Path

from . import config

E = lambda repo, file, name, size, note: {"repo": repo, "file": file, "name": name, "size": size, "note": note}
RECENT = "Needs a recent llama.cpp build."
BUILTIN = [
    E("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", "Qwen3.5 2B, Q4_K_M", "1.3 GB",
      "Best first test. The original small model at the same 4-bit size an export uses, so before/after is a fair comparison. " + RECENT),
    E("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q8_0.gguf", "Qwen3.5 2B, Q8_0", "2.0 GB",
      "Close to the original quality. Handy as an upper reference. " + RECENT),
    E("unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_K_M.gguf", "Gemma 4 E2B, Q4_K_M", "3.1 GB",
      "Google's small Gemma 4, 4-bit. " + RECENT),
    E("unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q8_0.gguf", "Gemma 4 E2B, Q8_0", "5.1 GB",
      "Gemma 4, 8-bit. Unsloth export of Gemma 4 may be limited to q8_0, so this is the fair baseline. " + RECENT),
    E("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", "Qwen3.5 4B, Q4_K_M", "",
      "Bigger Qwen3.5. The file name follows the 2B repo's pattern but was not individually checked; a wrong name fails with a clear message."),
    E("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf", "Qwen3.5 0.8B, Q4_K_M", "",
      "Tiny and very fast, good for checking your setup. File name follows the 2B pattern, not individually checked."),
]


def models_dir() -> Path:
    d = config.home() / "models"
    d.mkdir(exist_ok=True)
    return d


def examples() -> list[dict]:
    out = {(e["repo"], e["file"]): e for e in BUILTIN}
    f = config.home() / "gguf_examples.json"
    if f.exists():
        try:
            for e in json.loads(f.read_text(encoding="utf-8")):
                if isinstance(e, dict) and "repo" in e and "file" in e:
                    out[(e["repo"], e["file"])] = {"name": e["file"], "size": "", "note": "", **e}
        except (json.JSONDecodeError, TypeError):
            pass
    return list(out.values())


def local_ggufs() -> list[dict]:
    roots = [(models_dir(), "downloaded"), (config.runs_dir(), "exported")] + [
        (Path(p), "other") for p in os.environ.get("FTS_MODELS_DIR", "").split(os.pathsep) if p]
    seen, out = set(), []
    for root, where in roots:
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*.gguf")):
            key = str(f.resolve())
            if key not in seen:
                seen.add(key)
                out.append({"path": str(f), "name": f.name, "size_mb": round(f.stat().st_size / 1e6), "where": where})
    return out
