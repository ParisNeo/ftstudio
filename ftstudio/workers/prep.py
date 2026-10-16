"""Prepare a dataset. params:
  inputs?: [files .jsonl/.json/.csv/.tsv]
  hf_datasets?: ["owner/name[:config][@split][#limit]" or dict]   (Hugging Face; a limit streams just that many rows)
  format?: auto|messages|sharegpt|alpaca|prompt_completion|columns   (default auto, per record)
  columns?: {"user": col, "assistant": col, "system"?: col}          (for CSV or odd schemas)
  system_prompt?: system turn for records that have none; replace_system?=false: set true to overwrite existing system turns too
  tokenizer?: HF id or local path (enables chat-template rendering, token lengths, template probe)
  max_seq_len?: drop (never truncate) examples longer than this many tokens   (needs tokenizer)
  val_fraction?=0.05   dedupe?=true   decontaminate_against?: [custom eval .jsonl files]
Writes train.jsonl, val.jsonl ({"messages": [...]}), report.json, preview.txt, result.json."""
import json
import sys
from pathlib import Path

from ..data.prep import prepare


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if not getattr(tok, "chat_template", None):
        raise RuntimeError(f"tokenizer {name!r} has no chat_template; use the instruct/chat variant of the model")
    return tok


def main(run_dir: str):
    run = Path(run_dir)
    p = json.loads((run / "params.json").read_text())
    tok = load_tokenizer(p["tokenizer"]) if p.get("tokenizer") else None
    rep = prepare(p, run, tok)
    (run / "result.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("read", "kept", "train", "val", "dropped", "formats", "length_stats")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
