"""Turn raw documents into verified training entries with an LLM you supply (a callable `chat`).
Pipeline per passage: generate Q&A -> reject self-referential/short/duplicate/contaminated -> RUN the code in the answer
(repair with the error, up to N rounds) -> LLM judge checks every claim against the passage -> save to a library.
Nothing here is a guarantee of perfection: it is a filter. Review the dry run and rejects.jsonl before trusting a run."""
import json
import re
from pathlib import Path

from .. import library
from ..evals.custom import run_python
from .prep import _norm, blocked_prompts

GEN_SYSTEM = ("You write high-quality question and answer pairs for training a language model on a software library. "
              "You only use facts stated in the documentation excerpt you are given. You never invent functions, classes, "
              "parameters or behavior. If the excerpt has nothing worth teaching, you return [].")
GEN_PROMPT = """Documentation excerpt (from {source}):
\"\"\"
{chunk}
\"\"\"

Write {n} diverse question and answer pairs a developer might ask about this excerpt.
Rules:
- Use only facts stated in the excerpt. Do not invent names, parameters or behavior.
- Each question must stand alone: never say "the excerpt", "the text" or "above".
- Answers are precise and complete. When the excerpt shows how to use something, include a short complete runnable Python example in a ```python block.
- Code must import exactly what it uses and call only APIs that appear in the excerpt.
Return ONLY JSON: [{"question": "...", "answer": "..."}]"""
JUDGE_PROMPT = """You are a strict reviewer of training data.

Documentation excerpt:
\"\"\"
{chunk}
\"\"\"

Question: {question}
Answer: {answer}

Check: (1) every fact, name and parameter in the answer is stated in the excerpt, (2) nothing is invented, (3) the answer fully addresses the question.
Reply with exactly one line: PASS or FAIL: <short reason>"""
REPAIR_PROMPT = """The answer below contains Python code that failed when it was run.

Documentation excerpt:
\"\"\"
{chunk}
\"\"\"

Question: {question}
Answer:
{answer}

Error:
{error}

Rewrite the answer so that its code runs. Use only functions, classes and parameters that appear in the excerpt.
Return ONLY JSON: {"answer": "..."}"""

PYBLOCK = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.S)
SELF_REF = re.compile(r"\b(the|this) (excerpt|passage|text above|documentation provided)\b|\babove\b", re.I)


def _fill(t: str, **kw) -> str:
    for k, v in kw.items():
        t = t.replace("{" + k + "}", str(v))
    return t


def _join(x):
    return "\n".join(x) if isinstance(x, list) else (x or "")


def _clean(t: str) -> str:
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()


def split_text(text: str, size: int = 3000, overlap: int = 200) -> list[str]:
    if size <= overlap:
        raise ValueError("chunk_size must be larger than chunk_overlap")
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if len(para) > size:
            if cur.strip():
                out.append(cur)
                cur = ""
            out += [para[j:j + size] for j in range(0, len(para), size - overlap)]
            continue
        if len(cur) + len(para) + 2 > size and cur.strip():
            out.append(cur)
            cur = cur[-overlap:] if overlap else ""
        cur += ("\n\n" if cur else "") + para
    if cur.strip():
        out.append(cur)
    return [c.strip() for c in out if len(c.strip()) >= 40]


def iter_chunks(paths, size=3000, overlap=200, include=("*.md", "*.rst", "*.txt", "*.py")):
    files = []
    for s in paths:
        p = Path(s)
        if p.is_dir():
            for pat in include:
                files += sorted(p.rglob(pat))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"source not found: {s}")
    seen = set()
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, c in enumerate(split_text(text, size, overlap)):
            yield {"source": f.name, "path": str(f), "idx": i, "text": c}


def parse_pairs(text: str) -> list[dict]:
    t = _clean(text)
    a, b = t.find("["), t.rfind("]")
    for cand in ([t[a:b + 1]] if 0 <= a < b else []) + [t]:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = data.get("pairs") or data.get("items") or [data]
        if isinstance(data, list):
            return [{"question": d["question"].strip(), "answer": d["answer"].strip()} for d in data
                    if isinstance(d, dict) and isinstance(d.get("question"), str) and isinstance(d.get("answer"), str)
                    and d["question"].strip() and d["answer"].strip()]
    return []


def is_empty_list(text: str) -> bool:
    """The model answered with a valid empty JSON list (nothing worth teaching in this passage)."""
    t = _clean(text)
    a, b = t.find("["), t.rfind("]")
    try:
        return 0 <= a < b and json.loads(t[a:b + 1]) == []
    except json.JSONDecodeError:
        return False


def parse_answer(text: str):
    t = _clean(text)
    a, b = t.find("{"), t.rfind("}")
    try:
        d = json.loads(t[a:b + 1])
        return d["answer"].strip() if isinstance(d.get("answer"), str) and d["answer"].strip() else None
    except (json.JSONDecodeError, KeyError, AttributeError, ValueError):
        return None


def check_code(answer: str, must_import: str = "", setup: str = "", timeout: int = 20):
    """(ok, detail). Answers without a python block pass; blocks must run (exit 0) and import must_import."""
    blocks = PYBLOCK.findall(answer)
    if not blocks:
        return True, "no code"
    if must_import and not any(re.search(rf"^\s*(?:from|import)\s+{re.escape(must_import)}\b", b, re.M) for b in blocks):
        return False, f"code does not import {must_import}"
    for b in blocks:
        rc, _, err = run_python((setup + "\n" if setup else "") + b, timeout)
        if rc != 0:
            return False, f"exit {rc}: {err.strip()[-400:]}"
    return True, "ok"


def _jaccard(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / len(x | y) if x | y else 0.0


def run(p: dict, chat, out: Path, model: str = "local", log=print) -> dict:
    """chat(messages, temperature=, max_tokens=) -> str"""
    lib, mode, dry = p["library"], p.get("mode", "append"), bool(p.get("dry_run"))
    if mode not in ("append", "create"):
        raise ValueError("mode must be 'append' or 'create'")
    prompt = _join(p.get("prompt")) or GEN_PROMPT
    if "{chunk}" not in prompt:
        raise ValueError("your prompt must contain {chunk}, where the document passage goes")
    gen_system = _join(p.get("generator_system")) or GEN_SYSTEM
    n, rounds = int(p.get("per_chunk", 3)), int(p.get("repair_rounds", 1))
    judge_on, code_on = p.get("judge", True), p.get("verify_code", True)
    must_import, setup, timeout = p.get("must_import", ""), p.get("setup", ""), int(p.get("timeout", 20))
    max_chunks = int(p.get("max_chunks") or (3 if dry else 0))
    chunks = list(iter_chunks(p["sources"], int(p.get("chunk_size", 3000)), int(p.get("chunk_overlap", 200)),
                              tuple(p.get("include") or ("*.md", "*.rst", "*.txt", "*.py"))))
    if max_chunks:
        chunks = chunks[:max_chunks]
    if not chunks:
        raise ValueError("no readable text found in 'sources'")

    existing = library.read_all(lib) if mode == "append" or dry else []
    if mode == "create" and not dry:
        if library.read_all(lib) and not p.get("overwrite"):
            raise ValueError(f"library {lib!r} already has entries; use mode 'append' or set overwrite: true")
        library.reset(lib)
    qs = [_norm(next(m["content"] for m in e["messages"] if m["role"] == "user")) for e in existing]
    blocked = blocked_prompts(p.get("decontaminate_against") or [])
    target = out / "preview.jsonl" if dry else library.path(lib)
    if dry:
        target.write_text("")
    stats = {"chunks": 0, "generated": 0, "kept": 0, "rejected": {}, "repaired": 0}
    rej = open(out / "rejects.jsonl", "w", encoding="utf-8")

    def reject(reason, ch, q="", a="", detail=""):
        stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
        rej.write(json.dumps({"reason": reason, "source": ch["source"], "chunk": ch["idx"], "question": q, "answer": a,
                              "detail": detail}, ensure_ascii=False) + "\n")

    for ci, ch in enumerate(chunks, 1):
        stats["chunks"] += 1
        raw = _clean(chat([{"role": "system", "content": gen_system},
                           {"role": "user", "content": _fill(prompt, chunk=ch["text"], n=n, source=ch["source"])}],
                          temperature=float(p.get("temperature", 0.3)), max_tokens=int(p.get("max_tokens", 1800))))
        pairs = parse_pairs(raw)
        if not pairs and not is_empty_list(raw):
            reject("unparseable_output", ch, detail=raw[:200])
        kept_here = 0
        for pr in pairs[:n]:
            stats["generated"] += 1
            q, a = pr["question"], pr["answer"]
            if len(q) < 10 or len(a) < 20:
                reject("too_short", ch, q, a); continue
            if SELF_REF.search(q):
                reject("refers_to_excerpt", ch, q, a); continue
            nq = _norm(q)
            if nq in blocked:
                reject("contaminated", ch, q, a); continue
            if any(nq == x or _jaccard(nq, x) >= 0.85 for x in qs):
                reject("duplicate", ch, q, a); continue
            code_state = "none" if not PYBLOCK.search(a) else "ok"
            if code_on:
                ok, detail = check_code(a, must_import, setup, timeout)
                r = 0
                while not ok and r < rounds and not detail.startswith("code does not import"):
                    r += 1
                    fixed = parse_answer(chat([{"role": "system", "content": gen_system}, {"role": "user", "content": _fill(
                        REPAIR_PROMPT, chunk=ch["text"], question=q, answer=a, error=detail)}], temperature=0.2, max_tokens=1800))
                    if not fixed:
                        break
                    a = fixed
                    ok, detail = check_code(a, must_import, setup, timeout)
                    if ok:
                        stats["repaired"] += 1
                        code_state = "repaired"
                if not ok:
                    reject("code_fails" if not detail.startswith("code does not import") else "no_library_import", ch, q, a, detail)
                    continue
            verdict = "skipped"
            if judge_on:
                verdict = _clean(chat([{"role": "user", "content": _fill(JUDGE_PROMPT, chunk=ch["text"], question=q, answer=a)}],
                                      temperature=0.0, max_tokens=80))
                if not verdict.upper().startswith("PASS"):
                    reject("judge_rejected", ch, q, a, verdict[:200]); continue
                verdict = "PASS"
            msgs = ([{"role": "system", "content": _join(p["system_prompt"])}] if p.get("system_prompt") else []) + [
                {"role": "user", "content": q}, {"role": "assistant", "content": a}]
            library_entry = {"messages": msgs, "meta": {"source": ch["source"], "chunk": ch["idx"], "model": model,
                                                        "judge": verdict, "code": code_state}}
            with open(target, "a", encoding="utf-8") as f:
                f.write(json.dumps(library_entry, ensure_ascii=False) + "\n")
            qs.append(nq)
            stats["kept"] += 1
            kept_here += 1
        log(f"PASSAGE {ci}/{len(chunks)} {ch['source']}: kept {kept_here}, total kept {stats['kept']}")
    rej.close()
    return {**stats, "dry_run": dry, "library": lib, "target": str(target), "mode": mode, "model": model}
