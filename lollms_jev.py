#!/usr/bin/env python3
"""
lollms_jev — a small "System One" decision model: typed questions in, calibrated probabilities out.

Take a decoder LLM (default Qwen/Qwen3.5-0.8B), drop its LM head, and put typed heads on the
final hidden states:
    choice  -> pick 1 of N options (N varies per question)   softmax over per-option scores
    yesno   -> probability of "yes"                           sigmoid
    score   -> ordinal rubric 1..K (K <= 11)                  softmax over levels + EMD loss
Then LoRA fine-tune on (state, question, soft-target) rows kept in SQLite, and fit a temperature per
head type on held-out data so the confidence means something.

Design notes
  * Qwen3.5 is a hybrid (Gated DeltaNet + attention) model. DeltaNet layers are causal/recurrent, so
    custom 4D "parallel question" masks and bidirectional tricks do NOT apply. Instead:
      - each (state, question) is its own sequence; questions of one state are batched together;
      - causality is handled with a READOUT section: options are listed, then repeated after an
        <answer> marker, and the pooled hidden state is read from the repeated copy, so every
        option "sees" all the others.
  * Right padding only (padding never influences earlier tokens in a causal model).

Workflow
  pip install -U torch transformers peft datasets          # datasets only for import-cls
  python lollms_jev.py init     --db jev.db
  python lollms_jev.py import-cls --db jev.db --dataset ag_news --split train --text-cols text \
         --label-col label --question "Which topic is this text about?" --limit 3000
  python lollms_jev.py synth    --db jev.db --n 500 --teacher-model gpt-4o-mini      # invent states+schemas
  python lollms_jev.py distill  --db jev.db --k 5 --teacher-model gpt-4o-mini        # soft labels
  python lollms_jev.py train    --db jev.db --base Qwen/Qwen3.5-0.8B --out runs/jev0
  python lollms_jev.py predict  --ckpt runs/jev0 --input request.json

Teacher = lollms_client (pip install lollms-client). Configure the machine once, any binding works:
  python -m lollms_client.lollms_config_cli_env        # wizard -> ~/.lollms_client/config.yaml
or point at a LoLLMs server explicitly (default port 9642):
  python lollms_jev.py distill --db jev.db --teacher-binding lollms --teacher-host http://localhost:9642 \
         --teacher-model <model>            # add --teacher-key <key> if the server has security on
Training uses Unsloth by default (CUDA); falls back to plain HF+PEFT if it is missing/fails; --no-unsloth forces plain.
"""
import argparse, concurrent.futures as cf, hashlib, json, math, os, random, re, shutil, sqlite3, subprocess, sys, tempfile, time

_UNSLOTH_ERR = None
if len(sys.argv) > 1 and sys.argv[1] == "train" and "--no-unsloth" not in sys.argv:
    try:  # Unsloth must be imported before transformers/peft so it can patch them
        import unsloth  # noqa: F401
    except Exception as _u:  # noqa: BLE001  (no GPU / not installed / version clash)
        _UNSLOTH_ERR = _u
try:
    import torch, torch.nn as nn, torch.nn.functional as F
except Exception as _e:  # data / teacher commands still work without a usable torch
    torch, _TORCH_ERR = None, _e

MAX_LEVELS = 11
TYPES = ("choice", "score", "yesno")

# ----------------------------------------------------------------------------------------------
# Database (SQLite: zero-ops, ACID, resumable; targets are soft distributions stored as JSON)
# ----------------------------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS examples(
  id INTEGER PRIMARY KEY, state TEXT NOT NULL, source TEXT, hash TEXT UNIQUE, split TEXT);
CREATE TABLE IF NOT EXISTS questions(
  id INTEGER PRIMARY KEY,
  example_id INTEGER NOT NULL REFERENCES examples(id) ON DELETE CASCADE,
  type TEXT NOT NULL CHECK(type IN ('choice','score','yesno')),
  text TEXT NOT NULL, options TEXT, levels INTEGER,
  target TEXT,            -- JSON list: choice/score = probs, yesno = [p_yes]; NULL = needs distilling
  n_votes INTEGER DEFAULT 0,
  UNIQUE(example_id, type, text));
CREATE INDEX IF NOT EXISTS idx_q_target ON questions(target IS NULL);
"""


def db_connect(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


def split_for(state, val_pct):
    h = hashlib.sha1(state.encode()).hexdigest()
    return "val" if int(h[:8], 16) % 100 < val_pct else "train"  # split by STATE: no leakage


def norm_question(q):
    """Validate/normalise one question dict. Accepts `label` (hard) or `target` (soft)."""
    t = q["type"]
    if t not in TYPES:
        raise ValueError(f"bad type {t}")
    out = {"type": t, "text": str(q["text"]).strip(), "options": None, "levels": None, "target": None}
    if t == "choice":
        opts = [str(o) for o in q["options"]]
        if len(opts) < 2:
            raise ValueError("choice needs >= 2 options")
        out["options"], n = opts, len(opts)
    elif t == "score":
        K = int(q.get("levels", 5))
        if not 2 <= K <= MAX_LEVELS:
            raise ValueError(f"levels must be 2..{MAX_LEVELS}")
        out["levels"], n = K, K
    else:
        n = 1
    tgt, lab = q.get("target"), q.get("label")
    if tgt is not None:
        if t == "yesno":
            tgt = [float(tgt[0] if isinstance(tgt, (list, tuple)) else tgt)]
        else:
            if len(tgt) != n:
                raise ValueError("target length mismatch")
            s = float(sum(tgt))
            tgt = [float(x) / s for x in tgt]
    elif lab is not None:
        if t == "yesno":
            tgt = [1.0 if lab in (True, 1, "yes", "Yes", "true", "True") else 0.0]
        else:
            if t == "choice":
                idx = out["options"].index(lab) if isinstance(lab, str) and lab in out["options"] else int(lab)
            else:
                idx = int(lab) - 1  # score labels are 1-based
            if not 0 <= idx < n:
                raise ValueError("label out of range")
            tgt = [1.0 if i == idx else 0.0 for i in range(n)]
    out["target"] = tgt
    return out


def add_example(con, state, questions, source="", val_pct=8, group=None):
    state = state.strip()
    h = hashlib.sha1(state.encode()).hexdigest()
    row = con.execute("SELECT id FROM examples WHERE hash=?", (h,)).fetchone()
    if row:
        eid = row["id"]
    else:
        eid = con.execute("INSERT INTO examples(state,source,hash,split) VALUES(?,?,?,?)",
                          (state, source, h, split_for(group if group is not None else state, val_pct))).lastrowid
    n = 0
    for q in questions:
        q = norm_question(q)
        cur = con.execute(
            "INSERT OR IGNORE INTO questions(example_id,type,text,options,levels,target) VALUES(?,?,?,?,?,?)",
            (eid, q["type"], q["text"], json.dumps(q["options"]) if q["options"] else None, q["levels"],
             json.dumps(q["target"]) if q["target"] is not None else None))
        n += cur.rowcount
    return n


def row_to_q(r):
    return {"type": r["type"], "text": r["text"], "options": json.loads(r["options"]) if r["options"] else None,
            "levels": r["levels"]}


# ----------------------------------------------------------------------------------------------
# Teacher (lollms_client: one config, any binding — lollms server, ollama, openai, vllm, ...)
# ----------------------------------------------------------------------------------------------
TEACHER_CFG = {"binding": None, "host": None, "model": None, "key": None, "profile": None}
_TEACHER = None
_warned = set()


def get_teacher():
    global _TEACHER
    if _TEACHER is not None:
        return _TEACHER
    c = TEACHER_CFG
    try:
        from lollms_client import LollmsClient
        if c["binding"]:  # explicit: LollmsClient(llm_binding_name=..., llm_binding_config={...})
            conf = {k: v for k, v in (("host_address", c["host"]), ("model_name", c["model"]),
                                      ("service_key", c["key"])) if v}
            _TEACHER = LollmsClient(llm_binding_name=c["binding"], llm_binding_config=conf)
        else:  # centralised config: ~/.lollms_client/config.yaml, .env files, env vars
            from lollms_client.lollms_config_cli_env import get_client_from_env
            _TEACHER = get_client_from_env(run_wizard_if_fail=False)  # never block on the interactive wizard
        if c["profile"]:
            _TEACHER.switch_model(c["profile"])
        if getattr(_TEACHER, "llm", None) is None:  # LollmsClient swallows binding-load errors at init
            raise RuntimeError("LLM binding failed to load (missing dependency? the 'lollms' binding needs `pip install openai`)")
        from ascii_colors import ASCIIColors, LogLevel
        ASCIIColors.set_log_level(LogLevel.WARNING)  # lollms_client logs 2 INFO lines per call otherwise
    except Exception as e:  # noqa: BLE001
        sys.exit(f"teacher setup failed ({type(e).__name__}: {e}). Run `python -m lollms_client.lollms_config_cli_env` "
                 "once, or pass --teacher-binding/--teacher-host/--teacher-model.")
    return _TEACHER


def chat(system, prompt, temperature=0.7, max_tokens=800):
    """One teacher completion -> text ('' on failure). Thin wrapper over LollmsClient.generate_text."""
    for attempt in range(3):
        try:
            out = get_teacher().generate_text(prompt, system_prompt=system, n_predict=max_tokens,
                                              temperature=temperature, stream=False, think=False)
            if isinstance(out, dict):  # bindings return {"error"|"status":..., "message"|"error":...} on failure
                raise RuntimeError(out.get("message") or out.get("error") or str(out))
            return out or ""
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                if str(e) not in _warned:
                    _warned.add(str(e))
                    print("teacher error:", e, file=sys.stderr)
                return ""
            time.sleep(2 ** attempt)


def parse_json(s):
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


DOMAINS = ["customer support tickets", "email triage", "code review comments", "server logs / incidents",
           "sales leads", "HR / recruiting", "legal contracts", "medical admin messages (non-diagnostic)",
           "e-commerce reviews", "bug reports", "agent tool-selection", "chat moderation", "invoices / finance ops",
           "news articles", "scientific abstracts", "social media posts", "meeting transcripts", "product search queries",
           "browser-agent page states", "IoT sensor alerts", "education / student answers", "travel booking requests"]
LANGS = ["English"] * 6 + ["French", "Spanish", "German", "Arabic"]


def set_teacher(a):
    TEACHER_CFG.update(binding=a.teacher_binding, host=a.teacher_host, model=a.teacher_model,
                       key=a.teacher_key, profile=a.teacher_profile)


def cmd_synth(a):
    set_teacher(a)
    con = db_connect(a.db)
    rng = random.Random(a.seed)

    def one(_):
        dom, lang = rng.choice(DOMAINS), rng.choice(LANGS)
        length = rng.choice(["2-3 sentences", "a short paragraph", "a long, noisy, realistic message (150-300 words)",
                             "a JSON blob of application state", "a log excerpt"])
        prompt = f"""Invent ONE realistic application state for the domain "{dom}", written in {lang}, as {length}.
Then write 3 to 5 typed decision questions a piece of software might need answered about it. Mix types:
  {{"type":"choice","text":"...","options":["...","..."]}}   (2-8 options, options must be mutually exclusive)
  {{"type":"yesno","text":"..."}}
  {{"type":"score","text":"... (define the rubric: what 1 and K mean)","levels":5}}   (levels 3..10)
Questions must be answerable from the state alone; include at least one genuinely ambiguous question.
Reply with JSON only: {{"state":"...","questions":[...]}}"""
        return dom, parse_json(chat("You generate realistic, varied test data. Output JSON only.", prompt, temperature=1.0, max_tokens=1500))

    ok = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for dom, j in ex.map(one, range(a.n)):
            try:
                ok += bool(add_example(con, j["state"], j["questions"], f"synth:{dom}", a.val_pct))
            except Exception:  # noqa: BLE001
                continue
            con.commit()
    print(f"synthesised {ok}/{a.n} examples (targets empty — run `distill` next)")


def cmd_distill(a):
    set_teacher(a)
    con = db_connect(a.db)
    rows = con.execute("SELECT q.*, e.state FROM questions q JOIN examples e ON e.id=q.example_id "
                       "WHERE q.target IS NULL LIMIT ?", (a.limit,)).fetchall()
    rng = random.Random(a.seed)
    print(f"distilling {len(rows)} questions x {a.k} samples")

    def label(r):
        q = row_to_q(r)
        votes, valid = None, 0
        for _ in range(a.k):
            if q["type"] == "choice":
                perm = list(range(len(q["options"])))
                rng.shuffle(perm)  # kill the teacher's position bias
                shown = "\n".join(f"{i}. {q['options'][p]}" for i, p in enumerate(perm))
                ask = f"OPTIONS:\n{shown}\nReply JSON only: {{\"answer\": <index of the best option>}}"
            elif q["type"] == "yesno":
                perm, ask = None, 'Reply JSON only: {"answer": true} or {"answer": false}'
            else:
                perm, ask = None, f'Scale: integer 1 (lowest) to {q["levels"]} (highest). Reply JSON only: {{"answer": <int>}}'
            msg = f"STATE:\n{r['state']}\n\nQUESTION ({q['type']}): {q['text']}\n{ask}"
            j = parse_json(chat("You are a precise, calibrated annotator. Output JSON only.", msg,
                                temperature=a.temp, max_tokens=300))
            try:
                ans = j["answer"]
                if q["type"] == "choice":
                    idx = perm[int(ans)]
                    n = len(q["options"])
                elif q["type"] == "score":
                    idx, n = int(ans) - 1, q["levels"]
                    assert 0 <= idx < n
                else:
                    idx, n = (1 if ans in (True, "yes", "true") else 0), 2
                votes = votes or [0] * n
                votes[idx] += 1
                valid += 1
            except Exception:  # noqa: BLE001
                continue
        if valid < max(1, math.ceil(a.k / 2)):
            return r["id"], None, valid
        s = a.smooth
        if q["type"] == "yesno":
            return r["id"], [(votes[1] + s) / (valid + 2 * s)], valid
        return r["id"], [(v + s) / (valid + s * len(votes)) for v in votes], valid

    done = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for qid, tgt, nv in ex.map(label, rows):
            if tgt is not None:
                con.execute("UPDATE questions SET target=?, n_votes=? WHERE id=?", (json.dumps(tgt), nv, qid))
                done += 1
                if done % 25 == 0:
                    con.commit()
    con.commit()
    print(f"labelled {done}/{len(rows)}")


# ----------------------------------------------------------------------------------------------
# Ingest helpers
# ----------------------------------------------------------------------------------------------
def cmd_add(a):
    con = db_connect(a.db)
    n = 0
    for path in a.files:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                j = json.loads(line)
                n += add_example(con, j["state"], j["questions"], j.get("source", os.path.basename(path)), a.val_pct)
    con.commit()
    print("added", n, "questions")


def cmd_import_cls(a):
    """HF dataset -> typed questions. kind: choice (label = class), score (label = class+1), yesno (label in --yes-labels)."""
    from datasets import load_dataset  # pip install datasets
    ds = load_dataset(a.dataset, a.config, split=a.split)
    col = a.label_col
    names = getattr(ds.features[col], "names", None)  # ClassLabel -> readable names, int labels
    if a.limit:
        ds = ds.shuffle(seed=a.seed).select(range(min(a.limit, len(ds))))
    to_idx = None
    if names is None and a.kind not in ("yesno", "mc"):  # plain ints/strings: sort natively, never as strings
        uniq = sorted(set(ds[col]))
        names, to_idx = [str(u) for u in uniq], {u: i for i, u in enumerate(uniq)}
    yes = set(a.yes_labels)
    con = db_connect(a.db)
    n = 0
    for r in ds:
        cols = a.text_cols
        state = str(r[cols[0]]) if len(cols) == 1 else "\n".join(f"{c}: {r[c]}" for c in cols)
        lab, qtext = r[col], (str(r[a.question_col]) if a.question_col else a.question)
        if a.kind == "mc":  # options differ per row (MMLU, HellaSwag...): this is what teaches variable-N choice
            opts = [str(x) for x in r[a.options_col]]
            if len(opts) < 2 or int(lab) < 0:
                continue
            q = {"type": "choice", "text": qtext, "options": opts, "label": int(lab)}
        elif a.kind == "yesno":
            q = {"type": "yesno", "text": qtext, "label": str(lab) in yes or bool(names and isinstance(lab, int) and names[lab] in yes)}
        else:
            idx = to_idx[lab] if to_idx else int(lab)
            if idx < 0:
                continue  # unlabeled rows (-1)
            q = ({"type": "choice", "text": qtext, "options": names, "label": idx} if a.kind == "choice"
                 else {"type": "score", "text": qtext, "levels": len(names), "label": idx + 1})
        n += add_example(con, state, [q], f"cls:{a.dataset}", a.val_pct)
    con.commit()
    print(f"imported {n} {a.kind} questions from {a.dataset}")


def cmd_import_jev(a):
    """Jev-format decision datasets (e.g. MaziyarPanahi/Jev-RAG-Claim-Verification-130K): typed question + per-label
    criteria + JSON state + human gold label AND Jev's own probabilities. Splits are made by CONTEXT, so no document
    leaks across train/val (the dataset's own splits share 349 of 2,019 contexts)."""
    from datasets import load_dataset, concatenate_datasets
    parts = [load_dataset(a.dataset, split=sp) for sp in a.splits]
    ds = concatenate_datasets(parts)
    need = {"question_text", "criteria_json", "state_json", "primitive", "gold_label", "gold_score"}
    if need - set(ds.column_names):
        sys.exit(f"{a.dataset} is not Jev-format: missing {sorted(need - set(ds.column_names))}")
    if a.limit:
        ds = ds.shuffle(seed=a.seed).select(range(min(a.limit, len(ds))))
    con = db_connect(a.db)
    n = skipped = 0
    for r in ds:
        if r.get("question_key") in a.skip_keys or r["primitive"] not in ("choice", "noul"):
            skipped += 1
            continue
        state, crit = json.loads(r["state_json"]), json.loads(r["criteria_json"])
        text = r["question_text"] + "\nCriteria:\n" + "\n".join(f"- {k}: {v}" for k, v in crit.items())
        sstr = "\n".join(f"{k}: {v}" for k, v in state.items())
        group = str(state.get("authoritative_context") or r.get("group_id") or sstr)
        pj = r.get("jev_prediction")
        if r["primitive"] == "noul":
            g = float(r["gold_score"] if r["gold_score"] is not None else r["gold_label"] == "true")
            j = float(pj) if pj not in (None, "") else g
            q = {"type": "yesno", "text": text, "target": {"gold": g, "jev": j, "mix": (g + j) / 2}[a.target]}
        else:
            opts = list(crit)
            if r["gold_label"] not in opts:
                skipped += 1
                continue
            g = [1.0 if o == r["gold_label"] else 0.0 for o in opts]
            try:
                pr = json.loads(r.get("jev_probabilities_json") or "{}")
                j = [float(pr.get(o, 0.0)) for o in opts]
            except (ValueError, TypeError):
                j = [0.0] * len(opts)
            j = [x / sum(j) for x in j] if sum(j) > 0 else g
            q = {"type": "choice", "text": text, "options": opts,
                 "target": {"gold": g, "jev": j, "mix": [(x + y) / 2 for x, y in zip(g, j)]}[a.target]}
        n += add_example(con, sstr, [q], f"jev:{a.dataset.split('/')[-1]}", a.val_pct, group=group)
    con.commit()
    print(f"imported {n} questions from {a.dataset} (skipped {skipped}); target={a.target}")


def cmd_stats(a):
    con = db_connect(a.db)
    for r in con.execute("SELECT e.split, q.type, q.target IS NOT NULL AS labelled, COUNT(*) n FROM questions q "
                         "JOIN examples e ON e.id=q.example_id GROUP BY 1,2,3 ORDER BY 1,2,3"):
        print(dict(r))


# ----------------------------------------------------------------------------------------------
# Input construction (the "readout" trick lives here)
# ----------------------------------------------------------------------------------------------
def _enc(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def build_inputs(tok, state, q, max_len=1024, opt_tok=48):
    """-> (input_ids, pooled positions). choice: one position per option; yesno/score: one position."""
    t = q["type"]
    if t == "choice":  # shrink per-option budget when there are many options
        opt_tok = max(4, min(opt_tok, (max_len // 2) // (2 * len(q["options"]))))
    qhead = _enc(tok, f"\n</state>\n<question type={t}>\n{q['text']}\n")[:256]
    mid = []
    if t == "choice":
        mid = _enc(tok, "Options:")
        for i, o in enumerate(q["options"]):
            mid += _enc(tok, f"\n[{i}] {o}")[:opt_tok]
    elif t == "score":
        mid = _enc(tok, f"Rate 1 (lowest) to {q['levels']} (highest).")
    else:
        mid = _enc(tok, "Answer yes or no.")
    ans = _enc(tok, "\n<answer>")
    segs = []
    if t == "choice":
        segs = [_enc(tok, f"\n[{i}] {o}")[:opt_tok] for i, o in enumerate(q["options"])]
    shead = _enc(tok, "<state>\n")
    fixed = len(shead) + len(qhead) + len(mid) + len(ans) + sum(len(s) for s in segs)
    budget = max(16, max_len - fixed)
    sids = _enc(tok, state)
    if len(sids) > budget:  # keep head and tail of long states
        h = budget * 2 // 3
        sids = sids[:h] + sids[-(budget - h):]
    ids = shead + sids + qhead + mid + ans
    pos = [len(ids) - 1]
    if t == "choice":
        pos = []
        for s in segs:
            ids += s
            pos.append(len(ids) - 1)
    return ids, pos


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def load_backbone(name, dtype):
    from transformers import AutoModelForCausalLM
    full = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    bb = getattr(full, "model", None) or full.get_decoder()
    if hasattr(bb, "language_model"):  # composite VL wrapper -> text decoder only
        bb = bb.language_model
    hidden = bb.get_input_embeddings().embedding_dim
    del full  # <- the LM head ("output layer") is gone here; tied embeddings survive inside `bb`
    return bb, hidden


if torch is not None:
    def _head(h, out):
        return nn.Sequential(nn.Linear(h, h // 4), nn.GELU(), nn.Linear(h // 4, out))

    class JevModel(nn.Module):
        def __init__(self, backbone, hidden):
            super().__init__()
            self.backbone = backbone
            self.choice, self.yesno, self.score = _head(hidden, 1), _head(hidden, 1), _head(hidden, MAX_LEVELS)

        def forward(self, ids, mask, items):
            h = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            outs = []
            for b, it in enumerate(items):
                x = h[b, it["pos"]].float()
                t = it["q"]["type"]
                if t == "choice":
                    z = self.choice(x).squeeze(-1)
                elif t == "yesno":
                    z = self.yesno(x[-1:]).squeeze(-1)
                else:
                    z = self.score(x[-1])[: it["q"]["levels"]]
                outs.append(z)
            return outs


    def make_model_unsloth(base, lora_r, dtype, grad_ckpt, max_len):
        """UNTESTED here (needs a CUDA GPU): Unsloth loader + LoRA + 'unsloth' checkpointing, then the
        decoder is pulled out of the patched CausalLM exactly like load_backbone does (LM head dropped)."""
        from unsloth import FastModel
        m, _ = FastModel.from_pretrained(base, max_seq_length=max_len, dtype=dtype, load_in_4bit=False)
        kw = dict(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
                  use_gradient_checkpointing="unsloth" if grad_ckpt else False)
        try:
            m = FastModel.get_peft_model(m, target_modules="all-linear", **kw)
        except Exception:  # noqa: BLE001  (Unsloth build that only takes explicit module names)
            m = FastModel.get_peft_model(m, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                            "gate_proj", "up_proj", "down_proj"], **kw)
        inner = m.base_model.model  # PeftModel -> LoraModel -> the (patched) CausalLM
        bb = getattr(inner, "model", None) or inner.get_decoder()
        if hasattr(bb, "language_model"):
            bb = bb.language_model
        return JevModel(bb, bb.get_input_embeddings().embedding_dim)


    def make_model(base, lora_r, dtype, grad_ckpt, unsloth=False, max_len=1024):
        if unsloth:
            return make_model_unsloth(base, lora_r, dtype, grad_ckpt, max_len)
        bb, hidden = load_backbone(base, dtype)
        if grad_ckpt:
            bb.gradient_checkpointing_enable()
            bb.enable_input_require_grads()
        if lora_r > 0:
            from peft import LoraConfig, get_peft_model
            bb = get_peft_model(bb, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                               target_modules="all-linear"))
        return JevModel(bb, hidden)


    _PEFT = "backbone.base_model.model."

    @torch.no_grad()
    def fold_temperature(model, temps):
        for name, head in (("choice", model.choice), ("yesno", model.yesno), ("score", model.score)):
            T = float(temps.get(name, 1.0))
            if T != 1.0:  # logits/T == last Linear scaled by 1/T (softmax/sigmoid come after it)
                head[-1].weight.div_(T)
                head[-1].bias.div_(T)


    def trainable_state(model, canon=False):
        # canon=True strips the PEFT wrapper prefix so plain and Unsloth checkpoints share key names
        return {(("backbone." + n[len(_PEFT):]) if canon and n.startswith(_PEFT) else n): p.detach().cpu().clone()
                for n, p in model.named_parameters() if p.requires_grad}


    def item_loss(z, tgt, t):
        tgt = torch.tensor(tgt, device=z.device, dtype=torch.float32)
        if t == "yesno":
            return F.binary_cross_entropy_with_logits(z, tgt)
        lp = F.log_softmax(z, -1)
        loss = -(tgt * lp).sum()
        if t == "score":  # ordinal: squared EMD between CDFs
            loss = loss + ((lp.exp().cumsum(-1) - tgt.cumsum(-1)) ** 2).sum()
        return loss


    def collate(batch, pad_id, device):
        L = max(len(b["ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):  # RIGHT padding
            ids[i, : len(b["ids"])] = torch.tensor(b["ids"])
            mask[i, : len(b["ids"])] = 1
        return ids.to(device), mask.to(device)


def probs_from_logits(z, t, T=1.0):
    z = z.float() / T
    if t == "yesno":
        return [float(torch.sigmoid(z)[0])]
    return F.softmax(z, -1).tolist()


def load_items(con, split, tok, max_len, labelled=True, source=None, exclude=None):
    sql = ("SELECT e.state, e.source, q.* FROM questions q JOIN examples e ON e.id=q.example_id WHERE (e.split=? OR ?='all')"
           + (" AND q.target IS NOT NULL" if labelled else ""))
    items = []
    for r in con.execute(sql, (split, split)):
        src = r["source"] or ""
        if (source and not any(x in src for x in source)) or (exclude and any(x in src for x in exclude)):
            continue
        q = row_to_q(r)
        ids, pos = build_inputs(tok, r["state"], q, max_len)
        items.append({"ids": ids, "pos": pos, "q": q, "state": r["state"], "source": src, "target": json.loads(r["target"]) if r["target"] else None})
    return items


@torch.no_grad() if torch is not None else (lambda f: f)
def run_inference(model, items, pad_id, device, bs=16):
    """-> list of raw logit tensors (CPU, fp32), same order as items."""
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out = [None] * len(items)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = [items[i] for i in idx]
        ids, mask = collate(batch, pad_id, device)
        for i, z in zip(idx, model(ids, mask, batch)):
            out[i] = z.float().cpu()
    return out


# ----------------------------------------------------------------------------------------------
# Metrics + calibration
# ----------------------------------------------------------------------------------------------
def _padded(recs):
    n = max(len(r["z"]) for r in recs)
    Z = torch.full((len(recs), n), -1e9)
    T = torch.zeros((len(recs), n))
    for i, r in enumerate(recs):
        Z[i, : len(r["z"])] = r["z"]
        T[i, : len(r["target"])] = torch.tensor(r["target"])
    return Z, T


def nll_at(recs, t, temp):
    Z, T = _padded(recs)
    if t == "yesno":
        return float(F.binary_cross_entropy_with_logits(Z[:, 0] / temp, T[:, 0]))
    return float(-(T * F.log_softmax(Z / temp, -1)).sum(-1).mean())


def fit_temperature(recs, t):
    recs = [r for r in recs if r["type"] == t]
    if len(recs) < 20:
        return 1.0
    grid = [0.3 * (14 / 0.3) ** (i / 40) for i in range(41)]  # 0.3 .. 14
    best = min(grid, key=lambda T: nll_at(recs, t, T))
    if best in (grid[0], grid[-1]):
        print(f"warning: {t} temperature hit the search bound ({best:.2f}); val set too easy/small or model degenerate")
    fine = [best * (1 + (j - 10) * 0.02) for j in range(21)]
    return float(min(fine, key=lambda T: nll_at(recs, t, T)))


def auroc(ps, ys):
    idx = sorted(range(len(ps)), key=ps.__getitem__)
    ranks, i = [0.0] * len(ps), 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and ps[idx[j + 1]] == ps[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    npos = sum(ys)
    nneg = len(ys) - npos
    if not npos or not nneg:
        return None
    return (sum(r for r, y in zip(ranks, ys) if y) - npos * (npos + 1) / 2) / (npos * nneg)


def best_thr_acc(ps, ys):  # accuracy at the best cutoff (tuned on this very set: optimistic, shows threshold sensitivity)
    pairs = sorted(zip(ps, ys))
    n, pos_rest, neg_below = len(pairs), sum(ys), 0
    best = pos_rest / n
    for i, (p, y) in enumerate(pairs):
        neg_below += (not y)
        pos_rest -= y
        if i + 1 == n or pairs[i + 1][0] != p:
            best = max(best, (neg_below + pos_rest) / n)
    return best


def by_source(recs, temps, min_n=20):
    out = {}
    for src in sorted({r.get("source", "") for r in recs}):
        rs = [r for r in recs if r.get("source", "") == src]
        if len(rs) >= min_n:
            out[src or "(none)"] = metrics(rs, temps)
    return out


def metrics(recs, temps):
    res = {}
    for t in TYPES:
        rs = [r for r in recs if r["type"] == t]
        if not rs:
            continue
        T = temps.get(t, 1.0)
        conf, corr, mae, ps, ys = [], [], [], [], []
        for r in rs:
            p = probs_from_logits(r["z"], t, T)
            if t == "yesno":
                pred, tgt = int(p[0] > 0.5), int(r["target"][0] > 0.5)
                conf.append(max(p[0], 1 - p[0]))
                corr.append(pred == tgt)
                ps.append(p[0])
                ys.append(tgt)
            else:
                pred = max(range(len(p)), key=p.__getitem__)
                tgt = max(range(len(p)), key=r["target"].__getitem__)
                conf.append(p[pred])
                corr.append(pred == tgt)
                if t == "score":
                    mae.append(abs(pred - tgt))
        ece = 0.0
        for b in range(10):
            sel = [i for i, c in enumerate(conf) if b / 10 <= c < (b + 1) / 10 or (b == 9 and c == 1.0)]
            if sel:
                ece += len(sel) / len(rs) * abs(sum(corr[i] for i in sel) / len(sel) - sum(conf[i] for i in sel) / len(sel))
        res[t] = {"n": len(rs), "acc": sum(corr) / len(rs), "ece": ece, "nll": nll_at(rs, t, T),
                  **({"mae": sum(mae) / len(mae)} if mae else {})}
        if t == "yesno" and auroc(ps, ys) is not None:
            res[t]["auroc"], res[t]["acc_best_thr"] = auroc(ps, ys), best_thr_acc(ps, ys)
    return res


def show(tag, m):
    for t, v in m.items():
        print(f"[{tag}] {t:7s} " + " ".join(f"{k}={v[k]:.4f}" if isinstance(v[k], float) else f"{k}={v[k]}" for k in v))


def records_for(model, items, pad_id, device, bs):
    zs = run_inference(model, items, pad_id, device, bs)
    return [{"type": it["q"]["type"], "z": z, "target": it["target"], "source": it.get("source", "")} for it, z in zip(items, zs)]


# ----------------------------------------------------------------------------------------------
# Train / calibrate / eval / predict
# ----------------------------------------------------------------------------------------------
def pick_dtype(name):
    if name == "auto":
        name = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
    return getattr(torch, name)


def save_ckpt(out, model, cfg, temps):
    os.makedirs(out, exist_ok=True)
    from safetensors.torch import save_file  # no pickle: safe to publish on the Hub
    save_file({k: v.contiguous() for k, v in trainable_state(model, canon=True).items()}, os.path.join(out, "trainable.safetensors"))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    json.dump(temps, open(os.path.join(out, "calibration.json"), "w"), indent=2)


def load_ckpt(ckpt, device, dtype=None):
    from transformers import AutoTokenizer
    if not os.path.isdir(ckpt):  # "user/repo" on the Hugging Face Hub
        from huggingface_hub import snapshot_download
        ckpt = snapshot_download(ckpt)
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    dtype = dtype or pick_dtype(cfg.get("dtype", "auto"))
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    model = make_model(cfg["base"], cfg["lora_r"], dtype, False)
    st = os.path.join(ckpt, "trainable.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        sd = load_file(st)
    else:  # older checkpoints
        sd = torch.load(os.path.join(ckpt, "trainable.pt"))
    if cfg["lora_r"] > 0:  # canonical -> plain-PEFT key names
        sd = {("backbone.base_model.model." + k[len("backbone."):]) if k.startswith("backbone.") else k: v
              for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    temps = json.load(open(os.path.join(ckpt, "calibration.json")))
    return model.to(device).eval(), tok, cfg, temps


def cmd_train(a):
    from transformers import AutoTokenizer
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(a.dtype)
    tok = AutoTokenizer.from_pretrained(a.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    con = db_connect(a.db)
    hold = a.holdout_source or None  # sources never trained on: a real transfer test
    train, val = load_items(con, "train", tok, a.max_len, exclude=hold), load_items(con, "val", tok, a.max_len, exclude=hold)
    transfer = load_items(con, "all", tok, a.max_len, source=hold) if hold else []
    if not val:
        k = max(1, len(train) // 10)
        train, val = train[:-k], train[-k:]
        print("no val split in DB; holding out last", k, "train rows (may share states with train)")
    print(f"train={len(train)} val={len(val)} device={device} dtype={dtype}")
    grad_ckpt = a.grad_ckpt if a.grad_ckpt is not None else a.max_len > 2048  # long contexts: trade speed for memory
    use_unsloth = bool(a.unsloth) and _UNSLOTH_ERR is None and "unsloth" in sys.modules and device == "cuda"
    if a.unsloth and not use_unsloth:
        print(f"Unsloth unavailable ({_UNSLOTH_ERR or 'no CUDA device'}) -> plain HF+PEFT")
    try:
        model = make_model(a.base, a.lora_r, dtype, grad_ckpt, use_unsloth, a.max_len)
    except Exception as e:  # noqa: BLE001
        if not use_unsloth:
            raise
        import gc
        print(f"Unsloth path failed ({type(e).__name__}: {e}) -> falling back to plain HF+PEFT "
              "(if anything looks odd, rerun with --no-unsloth)")
        use_unsloth = False
        gc.collect(); torch.cuda.empty_cache()
        model = make_model(a.base, a.lora_r, dtype, grad_ckpt, False, a.max_len)
    model = model.to(device)
    head_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    bb_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    print(f"trainable: backbone(LoRA)={sum(p.numel() for p in bb_p)/1e6:.1f}M heads={sum(p.numel() for p in head_p)/1e6:.1f}M")
    opt = torch.optim.AdamW([{"params": bb_p, "lr": a.lr}, {"params": head_p, "lr": a.head_lr}], weight_decay=0.01)
    def make_batches(order):  # length-grouped; a batch never exceeds --bs rows or --max-tokens padded tokens
        out = []
        for s0 in range(0, len(order), a.bs * 50):
            cur, mx = [], 0
            for i in sorted(order[s0:s0 + a.bs * 50], key=lambda i: len(train[i]["ids"])):
                L = len(train[i]["ids"])
                if cur and (len(cur) >= a.bs or max(mx, L) * (len(cur) + 1) > a.max_tokens):
                    out.append(cur)
                    cur, mx = [], 0
                cur.append(i)
                mx = max(mx, L)
            if cur:
                out.append(cur)
        return out

    def augment(it):  # per-step option shuffling (kills position bias) + "none of the above" when the answer is removed
        q = it["q"]
        if q["type"] != "choice" or not (a.shuffle_options or a.reject_prob):
            return it
        opts, tgt = list(q["options"]), list(it["target"])
        if a.reject_prob and len(opts) >= 3 and max(tgt) > 0.9 and rng.random() < a.reject_prob:
            del opts[tgt.index(max(tgt))]
            opts.append("None of the above")
            tgt = [0.0] * (len(opts) - 1) + [1.0]
        if a.shuffle_options:
            perm = list(range(len(opts)))
            rng.shuffle(perm)
            opts, tgt = [opts[i] for i in perm], [tgt[i] for i in perm]
        q2 = {**q, "options": opts}
        ids, pos = build_inputs(tok, it["state"], q2, a.max_len)
        return {**it, "q": q2, "target": tgt, "ids": ids, "pos": pos}

    steps_per_epoch = math.ceil(len(make_batches(list(range(len(train))))) / a.accum)
    total = max(1, steps_per_epoch * a.epochs)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    cfg = {"base": a.base, "lora_r": a.lora_r, "max_len": a.max_len, "dtype": a.dtype, "hidden": model.score[0].in_features,
           "trained_with_unsloth": use_unsloth}

    best, best_loss, step = None, 1e9, 0
    t0, tokens, run_loss = time.time(), 0, []
    for ep in range(a.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        batches = make_batches(order)
        rng.shuffle(batches)
        model.train()
        for bi, idx in enumerate(batches):
            batch = [augment(train[i]) for i in idx]
            ids, mask = collate(batch, pad_id, device)
            zs = model(ids, mask, batch)
            loss = sum(item_loss(z, it["target"], it["q"]["type"]) for z, it in zip(zs, batch)) / len(batch)
            (loss / a.accum).backward()
            run_loss.append(float(loss))
            tokens += int(mask.sum())
            if (bi + 1) % a.accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(bb_p + head_p, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % max(1, min(10, total // 50)) == 0 or step == total:
                    el = time.time() - t0
                    hms = lambda x: f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{int(x % 60):02d}"  # noqa: E731
                    print(f"ep {ep} step {step}/{total} loss {sum(run_loss) / max(1, len(run_loss)):.4f} "
                          f"{tokens / el:.0f} tok/s ETA {hms(el / step * (total - step))} elapsed {hms(el)}")
                    run_loss = []
                if step % a.eval_every == 0 or step == total:
                    recs = records_for(model, val, pad_id, device, a.bs * 2)
                    vl = sum(nll_at([r], r["type"], 1.0) for r in recs) / len(recs)
                    print(f"== step {step} val nll {vl:.4f}")
                    if vl < best_loss:
                        best_loss, best = vl, trainable_state(model)
                    model.train()
    if best:
        model.load_state_dict(best, strict=False)
    recs = records_for(model, val, pad_id, device, a.bs * 2)
    show("val raw", metrics(recs, {}))
    temps = {t: fit_temperature(recs, t) for t in TYPES}
    show("val calibrated", metrics(recs, temps))
    print("temperatures:", temps)
    fold_temperature(model, temps)  # baked into the heads; calibration.json keeps 1.0 for compatibility
    cfg["temperature_folded"] = True
    save_ckpt(a.out, model, cfg, {t: 1.0 for t in TYPES})
    mets = {"val_raw": metrics(recs, {}), "val_calibrated": metrics(recs, temps), "by_source": by_source(recs, temps),
            "rows": {"train": len(train), "val": len(val), "transfer": len(transfer)},
            "sources": sorted({it.get("source", "") for it in train if it.get("source")}),
            "recipe": {"base": a.base, "lora_r": a.lora_r, "lr": a.lr, "epochs": a.epochs, "batch": a.bs * a.accum, "max_len": a.max_len}}
    for src, m in mets["by_source"].items():
        show(f"val {src}", m)
    if transfer:
        mets["transfer"] = metrics(records_for(model, transfer, pad_id, device, a.bs * 2), {})  # heads already calibrated
        show("TRANSFER (held-out sources, never trained)", mets["transfer"])
    json.dump(mets, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print("saved ->", a.out)


@torch.no_grad() if torch is not None else (lambda f: f)
def baseline_records(lm, tok, items, device, max_len):
    """The UNMODIFIED base LM reads the answer off its own logits (option letters, Yes/No, digits). No training involved."""
    tid = lambda x: _enc(tok, " " + x)[-1]  # noqa: E731
    recs = []
    for it in items:
        q, t = it["q"], it["q"]["type"]
        if t == "choice" and len(q["options"]) > 26:
            continue
        state = tok.decode(_enc(tok, it["state"])[: max(64, max_len - 300)])
        body = f"{state}\n\nQuestion: {q['text']}\n"
        if t == "choice":
            body += "".join(f"{chr(65 + i)}. {o}\n" for i, o in enumerate(q["options"])) + "Answer with the letter only.\nAnswer:"
        elif t == "yesno":
            body += "Answer Yes or No.\nAnswer:"
        else:
            body += f"Answer with a number from 1 (lowest) to {q['levels']} (highest).\nAnswer:"
        out = lm(input_ids=torch.tensor([_enc(tok, body)], device=device)).logits[0, -1].float().cpu()
        if t == "choice":
            z = out[[tid(chr(65 + i)) for i in range(len(q["options"]))]]
        elif t == "yesno":
            z = (out[tid("Yes")] - out[tid("No")]).view(1)
        else:
            z = out[[tid(str(k + 1)) for k in range(q["levels"])]]
        recs.append({"type": t, "z": z, "target": it["target"], "source": it["source"]})
    return recs


def cmd_baseline(a):
    """Zero-training reference. If a fine-tuned lollms_jev does not clearly beat this on the same rows, the heads are not earning their keep."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    lm = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=pick_dtype(a.dtype)).to(device).eval()
    items = load_items(db_connect(a.db), a.split, tok, a.max_len, source=a.source)[: a.limit or None]
    recs = baseline_records(lm, tok, items, device, a.max_len)
    show(f"baseline {a.base} (frozen, uncalibrated)", metrics(recs, {}))
    temps = {t: fit_temperature(recs, t) for t in TYPES}
    show("baseline (temperature fitted on these rows)", metrics(recs, temps))


def cmd_bench(a):
    """Official held-out splits (never trained on) of the datasets this model was built from: this model vs the frozen base, same rows."""
    import gc
    from transformers import AutoModelForCausalLM
    sp = a.specs or os.path.join(a.ckpt, "bench_specs.json")
    if not os.path.exists(sp):
        sys.exit("no bench_specs.json next to the checkpoint (the Easy tab writes it); pass --specs file.json")
    specs = [d for d in json.load(open(sp)) if d.get("eval_split")]
    tmp = os.path.join(tempfile.mkdtemp(prefix="jev_bench_"), "bench.db")
    done = {}
    for d in specs:
        try:
            cmd_import_cls(argparse.Namespace(db=tmp, dataset=d["id"], config=d.get("config"), split=d["eval_split"], text_cols=d["text"],
                                              label_col=d["label"], question=d.get("q") or "", question_col=d.get("qcol"), kind=d["kind"],
                                              yes_labels=d.get("yes") or ["1", "True", "true", "yes"], options_col=d.get("opts"),
                                              limit=a.limit, seed=0, val_pct=0))
            done[f"cls:{d['id']}"] = d["eval_split"]
        except Exception as e:  # noqa: BLE001
            print(f"bench: skipped {d['id']} ({type(e).__name__}: {str(e)[:100]})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, cfg, temps = load_ckpt(a.ckpt, device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    items = load_items(db_connect(tmp), "all", tok, cfg["max_len"])
    if not items:
        sys.exit("nothing to benchmark (no dataset with a configured held-out split could be loaded)")
    ours = by_source(records_for(model, items, pad_id, device, 8), temps, min_n=1)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    lm = AutoModelForCausalLM.from_pretrained(cfg["base"], torch_dtype=pick_dtype("auto")).to(device).eval()
    base = by_source(baseline_records(lm, tok, items, device, 2048), {}, min_n=1)
    out = {src: {"split": done.get(src, ""), "model": m, "base": base.get(src)} for src, m in ours.items()}
    json.dump({"limit": a.limit, "base": cfg["base"], "results": out}, open(os.path.join(a.ckpt, "bench.json"), "w"), indent=2)
    for src, r in out.items():
        show(f"BENCH {src} [{r['split']}] model", r["model"])
        if r["base"]:
            show(f"BENCH {src} [{r['split']}] frozen base", r["base"])


def cmd_calibrate(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, cfg, _ = load_ckpt(a.ckpt, device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    val = load_items(db_connect(a.db), "val", tok, cfg["max_len"])
    recs = records_for(model, val, pad_id, device, a.bs)
    temps = {t: fit_temperature(recs, t) for t in TYPES}
    show("before", metrics(recs, {}))
    show("after", metrics(recs, temps))
    fold_temperature(model, temps)
    cfg["temperature_folded"] = True
    save_ckpt(a.ckpt, model, cfg, {t: 1.0 for t in TYPES})


def cmd_eval(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, cfg, temps = load_ckpt(a.ckpt, device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    items = load_items(db_connect(a.db), a.split, tok, cfg["max_len"], source=a.source)
    recs = records_for(model, items, pad_id, device, a.bs)
    show(a.split, metrics(recs, temps))
    if a.by_source:
        for src, m in by_source(recs, temps).items():
            show(src, m)


def cmd_predict(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.onnx:
        sess, tok, cfg, temps = load_onnx(a.onnx, a.onnx_file)
    elif a.merged:
        model, tok, cfg, temps = load_merged(a.merged, device)
    else:
        model, tok, cfg, temps = load_ckpt(a.ckpt, device)
        if a.merge and hasattr(model.backbone, "merge_and_unload"):
            model.backbone = model.backbone.merge_and_unload()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    req = json.load(open(a.input) if a.input != "-" else sys.stdin)
    qs = [norm_question(q) for q in req["questions"]]
    items = []
    for q in qs:
        ids, pos = build_inputs(tok, req["state"], q, cfg["max_len"])
        items.append({"ids": ids, "pos": pos, "q": q})
    t0 = time.time()
    zs = run_onnx(sess, items) if a.onnx else run_inference(model, items, pad_id, device, 32)
    ms = (time.time() - t0) * 1000
    out = []
    for q, z in zip(qs, zs):
        p = probs_from_logits(z, q["type"], temps.get(q["type"], 1.0))
        if q["type"] == "choice":
            i = max(range(len(p)), key=p.__getitem__)
            out.append({"question": q["text"], "answer": q["options"][i], "confidence": p[i],
                        "probs": dict(zip(q["options"], p))})
        elif q["type"] == "yesno":
            out.append({"question": q["text"], "answer": p[0] > 0.5, "p_yes": p[0], "confidence": max(p[0], 1 - p[0])})
        else:
            i = max(range(len(p)), key=p.__getitem__)
            out.append({"question": q["text"], "score": i + 1, "expected": sum((k + 1) * x for k, x in enumerate(p)),
                        "confidence": p[i], "probs": p})
    print(json.dumps({"results": out, "latency_ms": round(ms, 1)}, indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------------------------------
# Export: merged HF model, ONNX (backbone + heads in one graph), GGUF (backbone; heads shipped beside it)
# ----------------------------------------------------------------------------------------------
def _load_sd(ckpt):
    st = os.path.join(ckpt, "trainable.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st)
    return torch.load(os.path.join(ckpt, "trainable.pt"))


def build_merged_full(ckpt, cfg):
    """The ORIGINAL causal LM (its own lm_head intact, so llama.cpp recognises it) + the trained LoRA merged in."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    full = AutoModelForCausalLM.from_pretrained(cfg["base"], torch_dtype=torch.float32)
    pm = get_peft_model(full, LoraConfig(r=cfg["lora_r"], lora_alpha=2 * cfg["lora_r"], target_modules="all-linear"))
    params = dict(pm.named_parameters())
    for k, v in _load_sd(ckpt).items():
        if k.startswith("backbone."):
            hit = [n for n in params if n.endswith("." + k[len("backbone."):])]
            if len(hit) != 1:
                raise RuntimeError(f"cannot map LoRA tensor {k} ({len(hit)} matches)")
            params[hit[0]].data.copy_(v.float())
    return pm.merge_and_unload()


def write_sidecar(ckpt, d):
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in _load_sd(ckpt).items() if not k.startswith("backbone.")}, os.path.join(d, "heads.safetensors"))
    shutil.copy(os.path.join(ckpt, "config.json"), os.path.join(d, "jev_config.json"))  # NOT config.json: that is the HF model's
    shutil.copy(os.path.join(ckpt, "calibration.json"), d)


if torch is not None:
    class OnnxWrap(nn.Module):
        """ids, mask, pool[B,P] -> raw outputs of all three heads at the P pooled positions."""
        def __init__(self, jm):
            super().__init__()
            self.jm = jm

        def forward(self, ids, mask, pool):
            h = self.jm.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            x = torch.gather(h, 1, pool.unsqueeze(-1).expand(-1, -1, h.size(-1))).float()
            return self.jm.choice(x).squeeze(-1), self.jm.yesno(x).squeeze(-1), self.jm.score(x)


@torch.no_grad() if torch is not None else (lambda f: f)
def export_onnx(jm, tok, cfg, d, quant=None):
    import copy
    jm = copy.deepcopy(jm).eval()
    if hasattr(jm.backbone, "merge_and_unload"):
        jm.backbone = jm.backbone.merge_and_unload()
    w = OnnxWrap(jm).eval()
    ids = torch.randint(0, max(8, min(len(tok), 1000)), (2, 12))
    ex = (ids, torch.ones_like(ids), torch.tensor([[11, 5, 7], [11, 0, 0]]))
    path = os.path.join(d, "model.onnx")
    kw = dict(input_names=["input_ids", "attention_mask", "pool"], output_names=["choice", "yesno", "score"],
              dynamic_axes={"input_ids": {0: "b", 1: "l"}, "attention_mask": {0: "b", 1: "l"}, "pool": {0: "b", 1: "p"},
                            "choice": {0: "b", 1: "p"}, "yesno": {0: "b", 1: "p"}, "score": {0: "b", 1: "p"}})
    try:
        torch.onnx.export(w, ex, path, opset_version=17, dynamo=False, **kw)
    except Exception as e1:  # noqa: BLE001
        try:
            torch.onnx.export(w, ex, path, opset_version=18, **kw)  # default exporter of newer torch
        except Exception as e2:  # noqa: BLE001
            raise RuntimeError(f"ONNX export failed ({type(e1).__name__}: {str(e1)[:200]} | {type(e2).__name__}: {str(e2)[:200]}). "
                               "Hybrid/linear-attention bases (Qwen3.5 DeltaNet layers) often cannot be traced; "
                               "a pure-attention base such as Qwen/Qwen3-0.6B exports reliably.") from e2
    try:  # parity check against torch
        import onnxruntime as ort, numpy as np
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"input_ids": ex[0].numpy(), "attention_mask": ex[1].numpy(), "pool": ex[2].numpy()})
        ref = w(*ex)
        print("onnx parity max|diff| per head:", [float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref)])
    except ImportError:
        print("pip install onnxruntime to verify the export")
    tok.save_pretrained(d)
    shutil.copy(os.path.join(os.path.dirname(d), "config.json"), os.path.join(d, "jev_config.json"))
    shutil.copy(os.path.join(os.path.dirname(d), "calibration.json"), d)
    if quant == "int8":  # dynamic int8 weights: ~4x smaller than fp32; check accuracy again before trusting it
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q8 = os.path.join(d, "model.int8.onnx")
        try:
            quantize_dynamic(path, q8, weight_type=QuantType.QInt8, use_external_data_format=True)
        except TypeError:
            quantize_dynamic(path, q8, weight_type=QuantType.QInt8)
        print(f"int8 model: {q8} ({os.path.getsize(q8) // 2**20} MB)")
    return path


def load_onnx(d, name="model.onnx"):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    cfgp = os.path.join(d, "jev_config.json")
    cfgp = cfgp if os.path.exists(cfgp) else os.path.join(d, "config.json")
    return (ort.InferenceSession(os.path.join(d, name), providers=prov), AutoTokenizer.from_pretrained(d),
            json.load(open(cfgp)), json.load(open(os.path.join(d, "calibration.json"))))


def load_merged(d, device):
    """Self-contained merged model (LoRA already folded in): no base model, no PEFT. d = local dir or Hub repo id."""
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    if not os.path.isdir(d):
        from huggingface_hub import snapshot_download
        d = os.path.join(snapshot_download(d, allow_patterns=["merged/*"]), "merged")
    elif os.path.isdir(os.path.join(d, "merged")):
        d = os.path.join(d, "merged")
    cfg = json.load(open(os.path.join(d, "jev_config.json")))
    bb, hidden = load_backbone(d, pick_dtype(cfg.get("dtype", "auto")))
    jm = JevModel(bb, hidden)
    res = jm.load_state_dict(load_file(os.path.join(d, "heads.safetensors")), strict=False)
    assert not res.unexpected_keys and all(k.startswith("backbone.") for k in res.missing_keys), res
    return jm.to(device).eval(), AutoTokenizer.from_pretrained(d), cfg, json.load(open(os.path.join(d, "calibration.json")))


def run_onnx(sess, items):
    import numpy as np
    out = []
    for it in items:  # one sequence per call: exact, no padding
        ids = np.array([it["ids"]], dtype=np.int64)
        c, y, s = sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids), "pool": np.array([it["pos"]], dtype=np.int64)})
        t = it["q"]["type"]
        out.append(torch.tensor(c[0] if t == "choice" else y[0, :1] if t == "yesno" else s[0, 0, : it["q"]["levels"]], dtype=torch.float32))
    return out


def export_gguf(merged_dir, d, quant, llama_cpp, quantize_bin=None):
    root = llama_cpp or os.environ.get("LLAMA_CPP_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "lollms_jev", "llama.cpp")
    script = os.path.join(root, "convert_hf_to_gguf.py")
    if not os.path.exists(script):
        print("cloning llama.cpp ->", root)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", root], check=True)
    first = quant if quant in ("f16", "bf16", "q8_0") else "f16"  # k-quants are made from an f16 file
    out = os.path.join(d, f"jev-{first}.gguf")
    try:
        subprocess.run([sys.executable, script, merged_dir, "--outfile", out, "--outtype", first], check=True)
    except subprocess.CalledProcessError:
        sys.exit(f"GGUF conversion failed. Install its deps first: pip install -r {root}/requirements/requirements-convert_hf_to_gguf.txt")
    if quant != first:
        exe = quantize_bin or shutil.which("llama-quantize") or next((x for x in (os.path.join(root, "build", "bin", "llama-quantize"),
                                                                              os.path.join(root, "build", "bin", "llama-quantize.exe")) if os.path.exists(x)), None)
        if not exe:
            sys.exit(f"{quant} needs the llama-quantize binary (build llama.cpp, or pass --llama-quantize). The f16 file is kept: {out}")
        final = os.path.join(d, f"jev-{quant}.gguf")
        subprocess.run([exe, out, final, quant.upper()], check=True)
        os.remove(out)
        out = final
    return out


def _size(path):
    tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)
    return f"{tot / 2**20:.0f} MB"


def write_card(ck, repo):
    cfg = json.load(open(os.path.join(ck, "config.json")))
    mp = os.path.join(ck, "metrics.json")
    m = json.load(open(mp)) if os.path.exists(mp) else {}
    have = {k: os.path.isdir(os.path.join(ck, k)) for k in ("merged", "onnx", "gguf")}

    def tbl(title, d):
        rows = "\n".join(f"| {t} | {v['n']} | {v['acc']:.3f} | {v['ece']:.3f} | {('%.3f' % v['auroc']) if 'auroc' in v else '-'} |" for t, v in d.items())
        return f"**{title}**\n\n| type | n | accuracy | ECE | AUROC |\n|---|---|---|---|---|\n{rows}\n\n"

    use = [f"- LoRA checkpoint (repo root, needs the base model): `python lollms_jev.py predict --ckpt {repo} --input request.json`"]
    if have["merged"]:
        use.append(f"- Merged model (`merged/`, self-contained, {_size(os.path.join(ck, 'merged'))}): `python lollms_jev.py predict --merged {repo} --input request.json`")
    if have["onnx"]:
        use.append(f"- ONNX (`onnx/`, {_size(os.path.join(ck, 'onnx'))}): download that folder, then `python lollms_jev.py predict --onnx onnx --input request.json` (`--onnx-file model.int8.onnx` for the int8 file if present)")
    if have["gguf"]:
        use.append("- GGUF (`gguf/`): the backbone for llama.cpp. Run it with embeddings and `--pooling none` to get per-token hidden states, then apply "
                   "`heads.safetensors` (choice / yesno / score heads, temperature already folded in). The glue code is not included.")
    bj, bench_md = os.path.join(ck, "bench.json"), ""
    if os.path.exists(bj):
        b = json.load(open(bj))
        rows = []
        for src, r in b["results"].items():
            for t, v in r["model"].items():
                bv = (r.get("base") or {}).get(t)
                rows.append(f"| {src.replace('cls:', '')} ({r['split']}) | {t} | {v['n']} | {v['acc']:.3f} | {('%.3f' % bv['acc']) if bv else '-'} | "
                            f"{('%.3f' % v['auroc']) if 'auroc' in v else '-'} | {v['ece']:.3f} |")
        bench_md = ("## Benchmarks\n\n| dataset (split) | type | n | this model | frozen base | AUROC | ECE |\n|---|---|---|---|---|---|---|\n"
                    + "\n".join(rows) + f"\n\nOfficial held-out splits of datasets used in training (their train splits only), at most {b['limit']} rows each. "
                    f"\"Frozen base\" is the unmodified `{b['base']}` answering from its own logits (option letters, Yes/No, digits) with no training, on the same rows. "
                    "Accuracy uses a 0.5 cutoff for yes/no; ECE is for this model only (its temperature was fitted on our own validation rows, not on these).\n\n")
    srcs = "\n".join(f"- {x}" for x in m.get("sources", [])) or "- (not recorded)"
    r = m.get("recipe", {})
    card = f"""---
base_model: {cfg['base']}
tags: [lollms-jev, decision-model, lora]
pipeline_tag: text-classification
---
# {repo.split('/')[-1]}

A small typed decision model: give it a state and typed questions (choice, yes/no, 1-K score) and it returns calibrated probabilities, not free text.
Built with `lollms_jev.py`: `{cfg['base']}` with LoRA (r={cfg['lora_r']}) and typed heads. Not affiliated with TypeSafe AI or its Jev model.

## Use
{chr(10).join(use)}

`request.json`: `{{"state": "...", "questions": [{{"type": "choice", "text": "...", "options": ["a", "b"]}}, {{"type": "yesno", "text": "..."}}, {{"type": "score", "text": "...", "levels": 5}}]}}`

## Results
""" + (tbl("Held-out rows (same sources as training; calibrated)", m["val_calibrated"]) if "val_calibrated" in m else "No metrics were recorded.\n\n") \
        + (tbl("Transfer: sources never trained on", m["transfer"]) if "transfer" in m else "No transfer set was held out for this run.\n\n") + f"""
Accuracy and ECE are measured against the dataset labels (gold or teacher) at a 0.5 cutoff for yes/no. Calibration temperatures are folded into the heads.

{bench_md}## Training
LoRA r={r.get('lora_r')}, lr {r.get('lr')}, {r.get('epochs')} epoch(s), batch {r.get('batch')}, max length {r.get('max_len')}. Data sources:
{srcs}

## Limits
The licence of the base model (`{cfg['base']}`) carries over to this fine-tune, and each training dataset has its own terms: check both before reuse. Outputs are only as good as the labels of those sources. Scores for sources not listed above are untested.
"""
    open(os.path.join(ck, "README.md"), "w", encoding="utf-8").write(card)


def cmd_publish(a):
    from huggingface_hub import HfApi
    write_card(a.ckpt, a.repo)
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)  # token comes from the environment, never from argv
    api.create_repo(a.repo, private=a.private, exist_ok=True)
    api.upload_folder(folder_path=a.ckpt, repo_id=a.repo, commit_message="lollms-jev checkpoint")
    api.upload_file(path_or_fileobj=os.path.abspath(__file__), path_in_repo="lollms_jev.py", repo_id=a.repo)
    print("published: https://huggingface.co/" + a.repo)


def cmd_export(a):
    jm, tok, cfg, _ = load_ckpt(a.ckpt, "cpu", torch.float32)
    merged, tmp = None, None
    for fmt in a.formats:
        d = os.path.join(a.ckpt, fmt)
        os.makedirs(d, exist_ok=True)
        if fmt == "onnx":
            print("exported", export_onnx(jm, tok, cfg, d, a.onnx_quant))
            continue
        if merged is None:
            merged = build_merged_full(a.ckpt, cfg)
            ids, _ = build_inputs(tok, "parity check text", {"type": "yesno", "text": "Is this a test?"}, cfg["max_len"])
            bb = getattr(merged, "model", merged)
            bb = getattr(bb, "language_model", bb)
            with torch.no_grad():
                diff = float((bb(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state
                              - jm.backbone(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state).abs().max())
            print(f"merged-model parity max|diff hidden| = {diff:.2e}" + ("  <-- too large, do not publish" if diff > 1e-3 else ""))
            merged = merged.to(getattr(torch, a.merged_dtype))  # parity was checked in fp32; ship the smaller dtype
        mdir = d if fmt == "merged" else (tmp := tempfile.mkdtemp(prefix="jev_merged_"))
        merged.save_pretrained(mdir)
        tok.save_pretrained(mdir)
        write_sidecar(a.ckpt, d)
        if fmt == "gguf":
            print("exported", export_gguf(mdir, d, a.quant, a.llama_cpp, a.llama_quantize))
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print("exported", d)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def P(name, fn, **kw):
        p = sp.add_parser(name, **kw)
        p.set_defaults(fn=fn)
        return p

    p = P("init", lambda a: (db_connect(a.db), print("ok")))
    p.add_argument("--db", default="jev.db")
    p = P("stats", cmd_stats); p.add_argument("--db", default="jev.db")
    p = P("add", cmd_add)
    p.add_argument("--db", default="jev.db"); p.add_argument("files", nargs="+"); p.add_argument("--val-pct", type=int, default=8)
    p = P("import-jev", cmd_import_jev)
    p.add_argument("--db", default="jev.db"); p.add_argument("--dataset", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"]); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--target", choices=["gold", "jev", "mix"], default="gold", help="human/source label, Jev's own probabilities, or their average")
    p.add_argument("--skip-keys", nargs="+", default=["quality"]); p.add_argument("--seed", type=int, default=0); p.add_argument("--val-pct", type=int, default=8)
    p = P("import-cls", cmd_import_cls)
    p.add_argument("--db", default="jev.db"); p.add_argument("--dataset", required=True); p.add_argument("--config")
    p.add_argument("--split", default="train"); p.add_argument("--text-cols", nargs="+", required=True)
    p.add_argument("--label-col", required=True); p.add_argument("--question", default="")
    p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-pct", type=int, default=8)
    p.add_argument("--kind", choices=["choice", "score", "yesno", "mc"], default="choice")
    p.add_argument("--options-col", help="kind=mc: column holding the list of options")
    p.add_argument("--question-col", help="take the question text from this column instead of --question")
    p.add_argument("--yes-labels", nargs="+", default=["1", "True", "true", "yes"], help="label values meaning YES (kind=yesno)")
    for name, fn in (("synth", cmd_synth), ("distill", cmd_distill)):
        p = P(name, fn)
        p.add_argument("--db", default="jev.db"); p.add_argument("--workers", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--teacher-binding", help="e.g. lollms, ollama, openai, vllm (default: your ~/.lollms_client config)")
        p.add_argument("--teacher-host", help="e.g. http://localhost:9642"); p.add_argument("--teacher-model")
        p.add_argument("--teacher-key", help="service key if the server has security on")
        p.add_argument("--teacher-profile", help="alias of a profile from your lollms_client config")
        if name == "synth":
            p.add_argument("--n", type=int, default=200); p.add_argument("--val-pct", type=int, default=8)
        else:
            p.add_argument("--k", type=int, default=5); p.add_argument("--temp", type=float, default=0.7)
            p.add_argument("--smooth", type=float, default=0.2); p.add_argument("--limit", type=int, default=100000)
    p = P("train", cmd_train)
    p.add_argument("--db", default="jev.db"); p.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--out", default="runs/jev0"); p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--bs", type=int, default=8); p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--lora-r", type=int, default=32); p.add_argument("--max-len", type=int, default=4096); p.add_argument("--max-tokens", type=int, default=16384, help="cap on padded tokens per micro-batch (memory guard)")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=None, help="save VRAM at ~30%% speed cost (default: on when --max-len > 2048)")
    p.add_argument("--unsloth", action=argparse.BooleanOptionalAction, default=True,
                   help="Unsloth loader + LoRA + checkpointing (default on; CUDA only; auto-falls back to plain HF+PEFT)")
    p.add_argument("--eval-every", type=int, default=100); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-options", action=argparse.BooleanOptionalAction, default=True, help="reshuffle choice options every step")
    p.add_argument("--reject-prob", type=float, default=0.05, help="chance of removing the right option and adding 'None of the above'")
    p.add_argument("--holdout-source", nargs="+", help="source substrings (e.g. cls:ag_news) kept out of training and reported as a transfer test")
    p = P("baseline", cmd_baseline)
    p.add_argument("--db", default="jev.db"); p.add_argument("--base", default="Qwen/Qwen3.5-0.8B"); p.add_argument("--split", default="val")
    p.add_argument("--source", nargs="+"); p.add_argument("--limit", type=int, default=500); p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p = P("bench", cmd_bench)
    p.add_argument("--ckpt", required=True); p.add_argument("--specs"); p.add_argument("--limit", type=int, default=300, help="rows per dataset")
    p = P("calibrate", cmd_calibrate)
    p.add_argument("--db", default="jev.db"); p.add_argument("--ckpt", required=True); p.add_argument("--bs", type=int, default=16)
    p = P("eval", cmd_eval)
    p.add_argument("--db", default="jev.db"); p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", help="train | val | all"); p.add_argument("--bs", type=int, default=16)
    p.add_argument("--source", nargs="+", help="only rows whose source contains one of these"); p.add_argument("--by-source", action="store_true")
    p = P("publish", cmd_publish)
    p.add_argument("--ckpt", required=True); p.add_argument("--repo", required=True, help="user/name"); p.add_argument("--private", action="store_true")
    p = P("export", cmd_export)
    p.add_argument("--ckpt", required=True); p.add_argument("--formats", nargs="+", choices=["merged", "onnx", "gguf"], required=True)
    p.add_argument("--quant", choices=["f16", "bf16", "q8_0", "q4_k_m", "q5_k_m", "q6_k"], default="q8_0", help="GGUF type; k-quants need llama-quantize")
    p.add_argument("--llama-cpp", help="llama.cpp checkout (default: auto-clone to ~/.cache/lollms_jev)"); p.add_argument("--llama-quantize", help="path to the llama-quantize binary")
    p.add_argument("--merged-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"); p.add_argument("--onnx-quant", choices=["int8"], help="also write model.int8.onnx")
    p = P("predict", cmd_predict)
    p.add_argument("--ckpt"); p.add_argument("--onnx", help="folder made by `export --formats onnx`"); p.add_argument("--onnx-file", default="model.onnx", help="e.g. model.int8.onnx")
    p.add_argument("--merged", help="folder or Hub repo id holding a merged model (no base/LoRA needed)"); p.add_argument("--input", default="-")
    p.add_argument("--merge", action="store_true", help="merge LoRA into the backbone for faster inference")
    a = ap.parse_args()
    if a.cmd in ("train", "calibrate", "eval", "predict", "export", "baseline", "bench") and torch is None:
        sys.exit(f"torch unavailable ({globals().get('_TORCH_ERR')}); pip install torch transformers peft")
    a.fn(a)


if __name__ == "__main__":
    main()
