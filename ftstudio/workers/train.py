"""QLoRA fine-tune with Unsloth on a finished prep job. params:
  dataset: {"job": prep_job_id} | {"dir": path}      base_model (HF id, 4-bit ok)
  max_seq_len?=2048 lora_r?=16 lora_alpha? learning_rate?=2e-4 epochs?=1 max_steps? batch_size?=1 grad_accum?=8 seed? eval_steps?
Loss is trained on assistant tokens only (anchors from the prep template probe).
Writes adapter/, metrics.jsonl, result.json. GPU libs are imported lazily."""
import json
import sys
import time
from pathlib import Path

from ..doctor import choose_backend, require_accelerator
from ..models import catalog_note
from ..train_utils import anchors, build_config, resolve_dataset, tokenizer_warning

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sft_config(SFTConfig, **kw):
    """TRL renamed args across versions; try the alternatives instead of dying."""
    alts = [{}, {"max_seq_length": "max_length"}, {"eval_strategy": "evaluation_strategy"},
            {"max_seq_length": "max_length", "eval_strategy": "evaluation_strategy"}]
    for alt in alts:
        try:
            return SFTConfig(**{alt.get(k, k): v for k, v in kw.items()})
        except TypeError as e:
            err = e
    raise err


def make_trainer(SFTTrainer, tok, **kw):
    """TRL renamed tokenizer= to processing_class=; try the new name first."""
    try:
        return SFTTrainer(processing_class=tok, **kw)
    except TypeError:
        return SFTTrainer(tokenizer=tok, **kw)


def main(run_dir: str):
    run = Path(run_dir)
    p = json.loads((run / "params.json").read_text())
    cfg, ds = build_config(p), resolve_dataset(p["dataset"])
    inst, resp = anchors(ds["report"])
    w = tokenizer_warning(ds["report"], cfg["base_model"])
    if w:
        print(w, flush=True)
    if catalog_note(cfg["base_model"]):
        print(catalog_note(cfg["base_model"]), flush=True)

    rep = require_accelerator(allow_cpu=bool(cfg.get("allow_cpu")))
    backend = choose_backend(cfg.get("backend", "auto"), rep, allow_cpu=bool(cfg.get("allow_cpu")))
    print(f"ENGINE: {backend} on {rep.get('accelerator') or rep.get('vendor')}", flush=True)
    if backend == "peft":
        from .train_peft import run_peft
        return run_peft(run, cfg, ds, inst, resp, rep)
    if backend == "mlx":
        from .train_mlx import run_mlx
        return run_mlx(run, cfg, ds, rep)
    try:
        from unsloth import FastLanguageModel, FastModel
        from unsloth.chat_templates import train_on_responses_only
    except Exception as e:  # installed but broken (e.g. mismatched trl/transformers versions)
        if cfg.get("backend", "auto") != "auto":
            raise
        print(f"WARNING: Unsloth is installed but cannot be imported ({e!r}). Falling back to the generic peft engine: slower and "
              "uses more memory. Run python install.py --repair to fix Unsloth.", flush=True)
        from .train_peft import run_peft
        return run_peft(run, cfg, ds, inst, resp, rep)
    from datasets import load_dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    import torch

    L = FastModel if cfg["loader"] == "model" else FastLanguageModel
    model, tok = L.from_pretrained(model_name=cfg["base_model"], max_seq_length=cfg["max_seq_len"],
                                   load_in_4bit=cfg["load_in_4bit"], dtype=None)
    ttok = getattr(tok, "tokenizer", tok)  # multimodal checkpoints return a processor; text work uses its tokenizer
    common = dict(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0, bias="none",
                  use_gradient_checkpointing="unsloth", random_state=cfg["seed"])
    if cfg["loader"] == "model":  # multimodal: LoRA on the language side only
        model = L.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                 finetune_attention_modules=True, finetune_mlp_modules=True, **common)
    else:
        model = L.get_peft_model(model, target_modules=TARGETS, **common)
    files = {"train": str(ds["train"])}
    has_val = ds["report"]["val"] > 0
    if has_val:
        files["val"] = str(ds["val"])
    data = load_dataset("json", data_files=files)
    fmt = lambda ex: {"text": ttok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}
    data = data.map(fmt, remove_columns=["messages"])

    metrics = []

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                rec = {"step": state.global_step, **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
                metrics.append(rec)
                print("METRIC " + json.dumps(rec), flush=True)

    bf16 = torch.cuda.is_bf16_supported()
    args = sft_config(SFTConfig, dataset_text_field="text", max_seq_length=cfg["max_seq_len"],
                      per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
                      warmup_ratio=0.03, num_train_epochs=cfg["epochs"], max_steps=cfg["max_steps"] or -1,
                      learning_rate=cfg["learning_rate"], bf16=bf16, fp16=not bf16, logging_steps=5,
                      optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear", seed=cfg["seed"],
                      output_dir=str(run / "ckpt"), report_to="none", save_strategy="no",
                      **({"eval_strategy": "steps", "eval_steps": cfg["eval_steps"]} if has_val else {}))
    trainer = make_trainer(SFTTrainer, ttok, model=model, train_dataset=data["train"],
                         eval_dataset=data["val"] if has_val else None, args=args, callbacks=[Log()])
    trainer = train_on_responses_only(trainer, instruction_part=inst, response_part=resp)

    # Show exactly what is being learned. Check this in the log: only assistant text should appear,
    # and BOS should not be doubled.
    ex = trainer.train_dataset[0]
    print("FIRST INPUT :", repr(ttok.decode(ex["input_ids"][:40])), flush=True)
    print("TRAINED ON  :", repr(ttok.decode([t for t in ex["labels"] if t != -100])[:300]), flush=True)

    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    model.save_pretrained(str(run / "adapter"))
    tok.save_pretrained(str(run / "adapter"))

    (run / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics))
    curve = [[m["step"], m["loss"]] for m in metrics if "loss" in m]
    step = max(1, len(curve) // 200)
    ev = [m["eval_loss"] for m in metrics if "eval_loss" in m]
    res = {"base_model": cfg["base_model"], "config": cfg, "backend": "unsloth", "accelerator": rep.get("accelerator"), "dataset_sha256": ds["report"]["sha256"],
           "adapter": str(run / "adapter"), "steps": metrics[-1]["step"] if metrics else 0,
           "final_loss": curve[-1][1] if curve else None, "eval_loss": ev[-1] if ev else None,
           "loss_curve": curve[::step], "eval_curve": [[m["step"], m["eval_loss"]] for m in metrics if "eval_loss" in m],
           "minutes": round(secs / 60, 1), "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}
    (run / "result.json").write_text(json.dumps(res, indent=2))
    print("done", json.dumps({k: res[k] for k in ("steps", "final_loss", "eval_loss", "minutes", "peak_vram_gb")}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
