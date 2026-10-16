"""Presets: named parameter sets the UI pours into a form. Built-ins below; add your own in
$FTS_HOME/presets.json as a list of {"id","kind","name","desc","params"} (same id overrides a built-in)."""
import json

from . import config

P = lambda id, kind, name, desc, params: {"id": id, "kind": kind, "name": name, "desc": desc, "params": params}

BUILTIN = [
    P("prep-tidy", "prep", "Tidy chat data", "Drops duplicates, broken rows and examples over 2048 tokens. Keeps 5% aside for checking.",
      {"max_seq_len": 2048, "val_fraction": 0.05, "dedupe": True}),
    P("prep-hf-alpaca", "prep", "Try it: 2000 examples from Alpaca cleaned (Hugging Face)", "Downloads only 2000 rows, so it is quick. A good way to test the whole pipeline with no data of your own.",
      {"hf_datasets": ["yahma/alpaca-cleaned#2000"], "max_seq_len": 2048, "val_fraction": 0.05}),
    P("prep-hf-finetome", "prep", "Try it: 5000 chat examples from FineTome (Hugging Face)", "High-quality conversations, streamed so only 5000 rows are downloaded.",
      {"hf_datasets": ["mlabonne/FineTome-100k#5000"], "max_seq_len": 2048, "val_fraction": 0.05}),
    P("prep-ascii-colors", "prep", "Your library: ascii_colors discussions (Hugging Face)", "ParisNeo/ascii_colors_discussions: 36 multi-turn conversations. Replaces its per-row system prompts (which say 'based only on this conversation') with one neutral expert prompt, so the knowledge goes into the model.",
      {"hf_datasets": ["ParisNeo/ascii_colors_discussions"], "tokenizer": "unsloth/Qwen3.5-2B", "system_prompt": 'You are an expert on the ascii_colors Python library by ParisNeo. Answer questions about its API accurately and give working code.', "replace_system": True, "max_seq_len": 2048, "val_fraction": 0.1}),
    P("prep-small", "prep", "Small dataset (under 500 examples)", "Keeps 10% aside so the check set is big enough to mean something.",
      {"max_seq_len": 2048, "val_fraction": 0.1}),
    P("prep-long", "prep", "Long examples (4096 tokens)", "Allows longer examples. Training them needs more GPU memory.",
      {"max_seq_len": 4096, "val_fraction": 0.05}),
    P("eval-quick", "eval", "Quick public check (math, 50 examples)",
      "Runs gsm8k on 50 examples: a fast sanity check, not a full benchmark. Also fill in the tokenizer.",
      {"suites": {"lm_eval": {"tasks": ["gsm8k"], "limit": 50, "num_fewshot": 5}}}),
    P("train-ascii-colors", "train", "Your library: ascii_colors on Qwen3.5 2B (tiny dataset)", "For about 36 conversations: bigger adapter, 4 passes, validation every 5 steps so you can see overfitting. Judge it with the ascii_colors benchmark, not by feel.",
      {"base_model": "unsloth/Qwen3.5-2B", "max_seq_len": 2048, "lora_r": 32, "epochs": 4, "learning_rate": 0.0002, "grad_accum": 4, "eval_steps": 5}),
    P("synth-dryrun", "synth", "Dry run: check the entries before a long run", "Processes only the first 3 passages and saves nothing to the library. Read preview.jsonl and rejects.jsonl, adjust your prompt, then run for real.",
      {"dry_run": True, "max_chunks": 3, "per_chunk": 3, "judge": True, "verify_code": True}),
    P("synth-code", "synth", "Library docs: entries whose code must run", "Every Python example is executed. Failures are repaired once with the error message, then dropped. A judge checks each answer against the passage. Fill in 'Code must import' with your module name.",
      {"per_chunk": 3, "judge": True, "verify_code": True, "repair_rounds": 1}),
    P("synth-concepts", "synth", "General docs: grounded Q&A (no code execution)", "For prose documents. A judge checks every answer against its passage.",
      {"per_chunk": 4, "judge": True, "verify_code": False}),
    P("train-smoke", "train", "Smoke test (minutes)", "10 training steps on short examples. Proves the whole chain works. The model will not be meaningfully better.",
      {"max_steps": 10, "max_seq_len": 1024, "lora_r": 8, "grad_accum": 1, "epochs": 1}),
    P("train-balanced", "train", "Balanced for a 12 GB GPU", "Good first real run: one pass over your data, 2048-token examples, adapter size 16.",
      {"max_seq_len": 2048, "lora_r": 16, "epochs": 1, "learning_rate": 0.0002, "grad_accum": 8}),
    P("train-thorough", "train", "Thorough (slower)", "Three passes, bigger adapter, 4096-token examples. Watch the validation loss: if it rises while training loss falls, it is memorizing.",
      {"max_seq_len": 4096, "lora_r": 32, "epochs": 3, "learning_rate": 0.0001, "grad_accum": 8}),
    P("train-qwen35-2b", "train", "Newest small model: Qwen3.5 2B", "Recent and fast, with lots of memory headroom. Trains in 16-bit LoRA (no 4-bit), set automatically.",
      {"base_model": "unsloth/Qwen3.5-2B", "max_seq_len": 2048, "lora_r": 16, "epochs": 1, "learning_rate": 0.0002, "grad_accum": 8}),
    P("train-qwen35-4b", "train", "Newest small model: Qwen3.5 4B (tight on 12 GB)", "Stronger, uses about 10 GB. Short examples (1024) keep it safe.",
      {"base_model": "unsloth/Qwen3.5-4B", "max_seq_len": 1024, "lora_r": 16, "epochs": 1, "learning_rate": 0.0002, "grad_accum": 8}),
    P("train-gemma4-e2b", "train", "Newest small model: Gemma 4 E2B", "Google's small Gemma 4, QLoRA, about 8 GB. Export to q8_0 for GGUF.",
      {"base_model": "unsloth/gemma-4-E2B-it", "max_seq_len": 2048, "lora_r": 16, "epochs": 1, "learning_rate": 0.0002, "grad_accum": 8}),
    P("train-lowvram", "train", "Low memory (8 GB GPU or out-of-memory errors)", "Shorter examples (1024) and a smaller adapter (8).",
      {"max_seq_len": 1024, "lora_r": 8, "epochs": 1, "grad_accum": 8}),
    P("export-everyday", "export", "Everyday (q4_k_m)", "Small file, good quality for most uses.", {"quantizations": ["q4_k_m"]}),
    P("export-quality", "export", "Higher quality (q8_0)", "Bigger file, closest to the original model.", {"quantizations": ["q8_0"]}),
    P("export-both", "export", "Both sizes", "Makes both files so you can compare them in the test step. Takes longer.", {"quantizations": ["q4_k_m", "q8_0"]}),
]


def all_presets() -> list[dict]:
    out = {p["id"]: p for p in BUILTIN}
    f = config.home() / "presets.json"
    if f.exists():
        try:
            for p in json.loads(f.read_text(encoding="utf-8")):
                if isinstance(p, dict) and {"id", "kind", "name", "params"} <= p.keys() and isinstance(p["params"], dict):
                    out[p["id"]] = {"desc": "", **p}
        except (json.JSONDecodeError, TypeError):
            pass
    return list(out.values())
