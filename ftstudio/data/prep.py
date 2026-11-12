"""Dataset preparation: load -> normalize -> decontaminate -> dedupe -> render/length-filter -> split -> write.
Pure functions + one orchestrator, so it is testable without a GPU or a worker."""
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from ..evals import custom
from .. import library
from . import hf
from .formats import RecordError, detect, normalize, to_messages


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_records(path: str):
    """Yield (line_no, record|None, error|None). .jsonl/.ndjson, .json (list or {"data":[..]}), .csv/.tsv"""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".jsonl", ".ndjson"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    r = json.loads(line)
                    yield (i, r, None) if isinstance(r, dict) else (i, None, "not a JSON object")
                except json.JSONDecodeError as e:
                    yield i, None, str(e)
    elif ext == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data") or data.get("examples") or []
        for i, r in enumerate(data, 1):
            yield (i, r, None) if isinstance(r, dict) else (i, None, "not a JSON object")
    elif ext in (".csv", ".tsv"):
        with open(p, newline="", encoding="utf-8-sig") as f:
            for i, r in enumerate(csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ","), 2):
                yield i, r, None
    else:
        raise ValueError(f"unsupported file type {ext!r} ({path}); use .jsonl .json .csv .tsv")


def blocked_prompts(files: list[str]) -> set[str]:
    out = set()
    for f in files:
        for it in custom.load(f):
            msgs = it.get("messages")
            out.add(_norm(msgs[-1]["content"] if msgs else it["prompt"]))
    return out


def render(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def ntokens(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def template_probe(tok) -> dict:
    """Render probe chats and split them. prefix|between|suffix come from a 2-turn chat; instruction_part and
    response_part (the user/assistant headers WITHOUT end-of-turn tokens, BOS or default system prompt) come
    from a 4-turn chat. These are the response-only masking anchors for training."""
    U = lambda t: {"role": "user", "content": t}
    A = lambda t: {"role": "assistant", "content": t}
    u, a, u2, a2 = "PROBE_USER_TEXT", "PROBE_ASSISTANT_TEXT", "PROBE_USER_TWO", "PROBE_ASSISTANT_TWO"
    text = render(tok, [U(u), A(a)])
    i, j = text.find(u), text.find(a)
    if i < 0 or j < 0:
        return {"rendered": text, "error": "template does not render message content verbatim"}
    out = {"rendered": text, "prefix": text[:i], "between": text[i + len(u):j], "suffix": text[j + len(a):]}
    try:
        t4 = render(tok, [U(u), A(a), U(u2), A(a2)])
        gap = t4[t4.index(a) + len(a):t4.index(u2)]
        eot = os.path.commonprefix([out["between"], out["suffix"]])
        out["response_part"] = out["between"][len(eot):]
        out["instruction_part"] = gap[len(os.path.commonprefix([gap, out["suffix"]])):]
    except ValueError:
        out["anchors_error"] = "could not derive instruction/response anchors from a 4-turn probe"
    return out


def _stats(xs):
    if not xs:
        return None
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
    return {"min": s[0], "mean": round(sum(s) / len(s), 1), "p50": q(0.5), "p95": q(0.95), "max": s[-1]}


def prepare(p: dict, out: Path, tok=None, hf_loader=None) -> dict:
    fmt_req = p.get("format", "auto")
    columns, max_len = p.get("columns"), p.get("max_seq_len")
    val_frac, dedupe = float(p.get("val_fraction", 0.05)), p.get("dedupe", True)
    if max_len and tok is None:
        raise ValueError("max_seq_len needs a tokenizer (token counts), none given")
    if fmt_req == "columns" and not columns:
        raise ValueError("format 'columns' needs a 'columns' mapping")
    blocked = blocked_prompts(p.get("decontaminate_against", []))

    drops, formats, examples, kept, seen, total = Counter(), Counter(), {}, [], set(), 0

    def drop(reason, path, line, detail=""):
        drops[reason] += 1
        ex = examples.setdefault(reason, [])
        if len(ex) < 3:
            ex.append({"file": path, "line": line, "detail": detail[:200]})

    sources = [(Path(f).name, load_records(f), fmt_req, columns) for f in p.get("inputs") or []]
    for name in p.get("libraries") or []:
        sources.append((f"library:{name}", load_records(str(library.path(name))), fmt_req, columns))
    for spec in p.get("hf_datasets") or []:
        sp = hf.parse_spec(spec)
        sources.append((hf.label(sp), hf.iter_records(sp, hf_loader), sp.get("format", fmt_req), sp.get("columns", columns)))
    if not sources:
        raise ValueError("no data: give 'inputs' (files), 'hf_datasets' (Hugging Face) and/or 'libraries'")
    per_source = {}

    for path, it, fmt_req, columns in sources:
        for line, rec, err in it:
            total += 1
            per_source[path] = per_source.get(path, 0) + 1
            if err:
                drop("parse_error", path, line, err)
                continue
            try:
                fmt = "columns" if columns and fmt_req in ("auto", "columns") else (
                    fmt_req if fmt_req != "auto" else detect(rec))
                if fmt is None:
                    raise RecordError("unknown_format", f"keys: {sorted(rec)[:8]}")
                msgs = normalize(to_messages(rec, fmt, columns), p.get("system_prompt"), bool(p.get("replace_system")))
            except RecordError as e:
                drop(e.reason, path, line, e.detail)
                continue
            if any(_norm(m["content"]) in blocked for m in msgs if m["role"] == "user"):
                drop("contaminated", path, line, "user prompt appears in an eval file")
                continue
            if dedupe:
                key = _sha(_norm(" ".join(m["role"] + ":" + m["content"] for m in msgs)))
                if key in seen:
                    drop("duplicate", path, line)
                    continue
                seen.add(key)
            n = len(json.dumps(msgs, ensure_ascii=False))
            if tok is not None:
                try:
                    n = ntokens(tok, render(tok, msgs))
                except Exception as e:  # template rejects some shapes (e.g. system role)
                    drop("template_error", path, line, repr(e))
                    continue
                if max_len and n > max_len:
                    drop("too_long", path, line, f"{n} tokens > {max_len}")  # dropped, never truncated
                    continue
            formats[fmt] += 1
            kept.append((msgs, n))

    def is_val(msgs):
        first_user = next(m["content"] for m in msgs if m["role"] == "user")
        return int(_sha("v1|" + _norm(first_user))[:8], 16) / 0xFFFFFFFF < val_frac

    train = [k for k in kept if not is_val(k[0])]
    val = [k for k in kept if is_val(k[0])]
    if val_frac > 0 and not val and len(train) >= 2:
        val.append(train.pop())  # tiny dataset: guarantee one val example

    h = hashlib.sha256()
    paths = {}
    for name, rows in (("train", train), ("val", val)):
        f = out / f"{name}.jsonl"
        with open(f, "w", encoding="utf-8") as fh:
            for msgs, _ in rows:
                line = json.dumps({"messages": msgs}, ensure_ascii=False)
                fh.write(line + "\n")
                h.update(line.encode())
        paths[name] = str(f)

    probe = template_probe(tok) if tok is not None else None
    training_ready = bool(
        probe and not probe.get("error") and not probe.get("anchors_error")
        and probe.get("instruction_part") and probe.get("response_part")
    )
    with open(out / "preview.txt", "w", encoding="utf-8") as fh:
        for i, (msgs, n) in enumerate(train[:5]):
            fh.write(f"===== sample {i} ({n} {'tokens' if tok else 'chars'}) =====\n")
            fh.write((render(tok, msgs) if tok is not None else json.dumps(msgs, ensure_ascii=False, indent=1)) + "\n\n")

    report = {
        "read": total, "sources": per_source, "kept": len(kept), "train": len(train), "val": len(val),
        "dropped": dict(drops), "drop_examples": examples, "formats": dict(formats),
        "length_unit": "tokens" if tok is not None else "chars",
        "length_stats": _stats([n for _, n in kept]),
        "max_seq_len": max_len, "template_probe": probe,
        "tokenizer": p.get("tokenizer"), "training_ready": training_ready,
        "sha256": h.hexdigest(), "paths": paths,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
