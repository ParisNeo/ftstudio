"""Pure helpers for train/export workers (no GPU imports, fully testable)."""
import json
from pathlib import Path

from . import config, db, models

DEFAULTS = {"max_seq_len": 2048, "lora_r": 16, "lora_alpha": None, "learning_rate": 2e-4, "epochs": 1,
            "max_steps": None, "load_in_4bit": True, "loader": "language", "backend": "auto", "dtype": "auto", "allow_cpu": False, "batch_size": 1, "grad_accum": 8, "seed": 3407, "eval_steps": 50}


def build_config(p: dict) -> dict:
    if not p.get("base_model"):
        raise ValueError("base_model is required (e.g. unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)")
    c = {**DEFAULTS, **models.defaults_for(p["base_model"]),
         **{k: v for k, v in p.items() if k in DEFAULTS and v is not None}, "base_model": p["base_model"]}
    c["lora_alpha"] = c["lora_alpha"] or c["lora_r"]
    for k in ("max_seq_len", "lora_r", "batch_size", "grad_accum"):
        if int(c[k]) < 1:
            raise ValueError(f"{k} must be >= 1")
    if c["loader"] not in ("language", "model"):
        raise ValueError("loader must be 'language' or 'model'")
    if c["learning_rate"] <= 0 or c["epochs"] <= 0:
        raise ValueError("learning_rate and epochs must be > 0")
    return c


def resolve_dataset(ds: dict) -> dict:
    """{"job": <succeeded prep job id>} or {"dir": <folder with train.jsonl,val.jsonl,report.json>}"""
    if "job" in ds:
        j = db.get_job(ds["job"])
        if j is None or j["kind"] != "prep" or j["status"] != "succeeded":
            raise ValueError(f"dataset job {ds['job']!r} is not a succeeded prep job")
        d = config.runs_dir() / ds["job"]
    elif "dir" in ds:
        d = Path(ds["dir"])
    else:
        raise ValueError("dataset needs 'job' or 'dir'")
    report = json.loads((d / "report.json").read_text())
    return {"train": d / "train.jsonl", "val": d / "val.jsonl", "report": report}


def anchors(report: dict) -> tuple[str, str]:
    pr = report.get("template_probe")
    if not pr:
        tok = report.get("tokenizer")
        if tok:
            raise ValueError(
                f"dataset has no template_probe despite being prepared with tokenizer {tok!r}. "
                "Re-run the prep job to regenerate it."
            )
        raise ValueError(
            "dataset was prepared without a tokenizer, so it has no template_probe and cannot be trained on. "
            "Re-run the prep job and set 'tokenizer' to the exact model you plan to train (e.g. unsloth/Qwen3.5-2B)."
        )
    if pr.get("error") or pr.get("anchors_error"):
        err = pr.get("error") or pr.get("anchors_error")
        raise ValueError(
            f"template probe generation failed during prep: {err}. Check that the tokenizer has a valid chat_template."
        )
    missing = [k for k in ("instruction_part", "response_part") if not pr.get(k)]
    if missing:
        raise ValueError(
            f"dataset probe is missing {', '.join(missing)}. The tokenizer chat_template likely does not properly separate "
            "the assistant's role marker from the generated text. Re-run prep with a proper instruct/chat model tokenizer."
        )
    return pr["instruction_part"], pr["response_part"]


def tokenizer_warning(report: dict, base_model: str):
    norm = lambda s: str(s).lower().split("/")[-1].replace("-bnb-4bit", "")
    t = report.get("tokenizer")
    if t and norm(t) != norm(base_model):
        return f"WARNING: dataset prepared with tokenizer {t!r} but training {base_model!r}; chat templates may differ"
    return None


def resolve_adapter(p: dict) -> Path:
    if p.get("train_job"):
        j = db.get_job(p["train_job"])
        if j is None or j["kind"] != "train" or j["status"] != "succeeded":
            raise ValueError(f"train job {p['train_job']!r} is not a succeeded train job")
        return config.runs_dir() / p["train_job"] / "adapter"
    if p.get("adapter_dir"):
        return Path(p["adapter_dir"])
    raise ValueError("export needs 'train_job' or 'adapter_dir'")


def queue_evals(ggufs: list[dict], suites: dict, name: str) -> list[str]:
    """One eval job per GGUF; the serial runner starts them after the export job exits."""
    return [db.create_job("eval", {"gguf": g["path"], "name": f"{name}-{g['quant']}", "suites": suites}) for g in ggufs]


def write_train_result(run, cfg, ds, metrics, secs, peak_gb, backend, accelerator, adapter, extra: dict | None = None) -> dict:
    """Same result.json schema for every training engine (the UI chart and export read it).
    `extra` lets a resumable run record metadata such as resumed_from / epochs_completed."""
    import json
    curve = [[m["step"], m["loss"]] for m in metrics if "loss" in m]
    ev = [[m["step"], m["eval_loss"]] for m in metrics if "eval_loss" in m]
    step = max(1, len(curve) // 200)
    res = {"base_model": cfg["base_model"], "config": cfg, "backend": backend, "accelerator": accelerator,
           "dataset_sha256": ds["report"]["sha256"], "adapter": str(adapter),
           "steps": metrics[-1]["step"] if metrics else 0, "final_loss": curve[-1][1] if curve else None,
           "eval_loss": ev[-1][1] if ev else None, "loss_curve": curve[::step], "eval_curve": ev,
           "minutes": round(secs / 60, 1), "peak_vram_gb": peak_gb}
    if extra:
        res.update(extra)
    (Path(run) / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics))
    (Path(run) / "result.json").write_text(json.dumps(res, indent=2))
    print("done", json.dumps({k: res[k] for k in ("backend", "steps", "final_loss", "eval_loss", "minutes", "peak_vram_gb")}), flush=True)
    return res
