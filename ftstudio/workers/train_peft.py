"""Generic LoRA fine-tuning with transformers + PEFT: NVIDIA (CUDA), AMD (ROCm), Intel (XPU), Apple (MPS), or CPU.
Slower and hungrier than Unsloth, but runs wherever PyTorch does. Loss on assistant tokens only (data/masking.py)."""
import time

from ..data.masking import tokenize_example
from ..train_utils import write_train_result

HEAVY = ("cuda", "rocm", "xpu")


def pick_dtype(requested: str, accel: str) -> str:
    """bf16 where the device supports it well, fp32 on MPS/CPU (fp16 LoRA training is unstable there). Override with dtype."""
    if requested and requested != "auto":
        if requested not in ("bf16", "fp16", "fp32"):
            raise ValueError("dtype must be auto, bf16, fp16 or fp32")
        return requested
    return "bf16" if accel in HEAVY else "fp32"


def make_args(TrainingArguments, **kw):
    """transformers renamed evaluation_strategy -> eval_strategy; try both."""
    for alt in ({}, {"eval_strategy": "evaluation_strategy"}):
        try:
            return TrainingArguments(**{alt.get(k, k): v for k, v in kw.items()})
        except TypeError as e:
            err = e
    raise err


def run_peft(run, cfg, ds, inst, resp, rep):
    import json

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments)

    accel = rep.get("accelerator") or "cpu"
    dtype_name = pick_dtype(cfg.get("dtype", "auto"), accel)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    print(f"peft engine: accelerator={accel}, dtype={dtype_name}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    kw = {"dtype": dtype} if int(__import__("transformers").__version__.split(".")[0]) >= 5 else {"torch_dtype": dtype}
    quant = None
    if cfg.get("load_in_4bit") and accel in ("cuda", "rocm"):
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        except Exception as e:
            print(f"4-bit loading unavailable ({e!r}); using 16-bit LoRA", flush=True)
    elif cfg.get("load_in_4bit"):
        print("4-bit (QLoRA) needs bitsandbytes on NVIDIA/AMD; using 16-bit LoRA instead. Memory use is higher.", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kw, **({"quantization_config": quant} if quant else {}))
    if quant:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    if accel in ("mps", "xpu"):
        model.to(accel)

    def encode(path):
        rows, dropped = [], 0
        for r in load_dataset("json", data_files=str(path), split="train"):
            ex = tokenize_example(tok, r["messages"], inst, resp, cfg["max_seq_len"])
            rows.append(ex) if ex else None
            dropped += ex is None
        return rows, dropped

    train, d1 = encode(ds["train"])
    has_val = ds["report"]["val"] > 0
    val, d2 = encode(ds["val"]) if has_val else ([], 0)
    if not train:
        raise RuntimeError("no trainable examples after masking; check the template probe in the prep report")
    print(f"examples: train {len(train)} (dropped {d1}), val {len(val)} (dropped {d2})", flush=True)
    first = train[0]
    print("TRAINED ON  :", repr(tok.decode([t for t in first["labels"] if t != -100])[:300]), flush=True)

    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids = torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch])
        lab = torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch])
        att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch])
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    metrics = []

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **k):
            if logs:
                rec = {"step": state.global_step, **{a: b for a, b in logs.items() if isinstance(b, (int, float))}}
                metrics.append(rec)
                print("METRIC " + json.dumps(rec), flush=True)

    args = make_args(TrainingArguments, output_dir=str(run / "ckpt"), per_device_train_batch_size=cfg["batch_size"],
                     gradient_accumulation_steps=cfg["grad_accum"], num_train_epochs=cfg["epochs"], max_steps=cfg["max_steps"] or -1,
                     learning_rate=cfg["learning_rate"], warmup_ratio=0.03, lr_scheduler_type="linear", logging_steps=5,
                     bf16=dtype_name == "bf16", fp16=dtype_name == "fp16", save_strategy="no", report_to=[],
                     remove_unused_columns=False, seed=cfg["seed"], use_cpu=accel == "cpu", optim="adamw_torch",
                     **({"eval_strategy": "steps", "eval_steps": cfg["eval_steps"]} if val else {}))
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val or None, data_collator=collate, callbacks=[Log()])
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    model.save_pretrained(str(run / "adapter"))
    tok.save_pretrained(str(run / "adapter"))
    peak = round(torch.cuda.max_memory_reserved() / 1e9, 2) if accel in ("cuda", "rocm") else None
    return write_train_result(run, cfg, ds, metrics, secs, peak, "peft", accel, run / "adapter")
