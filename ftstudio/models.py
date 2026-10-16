"""Model catalog the UI suggests, plus per-model training defaults.
Add your own in $FTS_HOME/models.json: a list of {"id","name","vram_gb","load_in_4bit","loader","notes"} (same id overrides).
vram_gb figures come from Unsloth's docs (Qwen3.5: bf16 LoRA; Gemma 4: QLoRA) and assume short sequences."""
import json

from . import config

M = lambda id, name, vram, q4, loader, notes, gguf=None: {
    "id": id, "name": name, "vram_gb": vram, "load_in_4bit": q4, "loader": loader, "notes": notes, "gguf_quants": gguf}
GEMMA_GGUF = ["q8_0", "bf16", "f16"]

BUILTIN = [
    M("unsloth/Qwen3.5-0.8B", "Qwen3.5 0.8B (2026)", 3, False, "language",
      "Tiny and very recent. Trains in 16-bit LoRA, no 4-bit: Unsloth advises against QLoRA on Qwen3.5. Needs transformers v5."),
    M("unsloth/Qwen3.5-2B", "Qwen3.5 2B (2026)", 5, False, "language",
      "Best first run on 12 GB: fast, lots of headroom. 16-bit LoRA. Needs transformers v5."),
    M("unsloth/Qwen3.5-4B", "Qwen3.5 4B (2026)", 10, False, "language",
      "Strongest small Qwen3.5 that fits 12 GB, but tight: keep examples short (1024). 16-bit LoRA. Needs transformers v5."),
    M("unsloth/gemma-4-E2B-it", "Gemma 4 E2B (2026)", 8, True, "model",
      "Google's small Gemma 4. QLoRA. Multimodal checkpoint: only the text side is trained here. GGUF export may be limited to q8_0/bf16/f16.", GEMMA_GGUF),
    M("unsloth/gemma-4-E4B-it", "Gemma 4 E4B (2026)", 10, True, "model",
      "Bigger Gemma 4. QLoRA (16-bit LoRA needs about 17 GB). Multimodal checkpoint: only the text side is trained. GGUF export may be limited to q8_0/bf16/f16.", GEMMA_GGUF),
    M("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "Llama 3.1 8B Instruct (proven)", 9, True, "language", "Older, well-trodden QLoRA path."),
    M("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "Qwen 2.5 7B Instruct (proven)", 9, True, "language", "Older, well-trodden QLoRA path."),
    M("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", "Mistral 7B Instruct v0.3 (proven)", 9, True, "language", "Older, well-trodden QLoRA path."),
]


def all_models() -> list[dict]:
    out = {m["id"]: m for m in BUILTIN}
    f = config.home() / "models.json"
    if f.exists():
        try:
            for m in json.loads(f.read_text(encoding="utf-8")):
                if isinstance(m, dict) and "id" in m:
                    out[m["id"]] = {**{"name": m["id"], "vram_gb": None, "load_in_4bit": True, "loader": "language",
                                       "notes": "", "gguf_quants": None}, **m}
        except (json.JSONDecodeError, TypeError):
            pass
    return list(out.values())


def find(model_id: str):
    return next((m for m in all_models() if m["id"] == model_id), None)


def defaults_for(model_id: str) -> dict:
    m = find(model_id)
    return {"load_in_4bit": m["load_in_4bit"], "loader": m["loader"]} if m else {}


def catalog_note(model_id: str):
    m = find(model_id)
    if not m:
        return None
    mode = "4-bit QLoRA" if m["load_in_4bit"] else "16-bit LoRA (no 4-bit)"
    return f"CATALOG: {m['name']}: {mode}, about {m['vram_gb']} GB GPU memory at short sequence lengths. {m['notes']}"


def quant_warning(model_id: str, quants: list[str]):
    m = find(model_id)
    allowed = (m or {}).get("gguf_quants")
    bad = [q for q in quants if allowed and q.lower() not in allowed]
    return f"WARNING: {model_id} GGUF export was reported to support only {allowed}; {bad} may fail" if bad else None
