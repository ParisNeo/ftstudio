"""Heterogeneous records -> canonical chat messages [{"role","content"}, ...]."""

ROLE_MAP = {
    "human": "user", "user": "user", "prompter": "user",
    "gpt": "assistant", "assistant": "assistant", "bot": "assistant", "model": "assistant", "chatgpt": "assistant",
    "system": "system",
}
PC_PAIRS = [("prompt", "completion"), ("prompt", "response"), ("question", "answer"),
            ("input", "output"), ("query", "response")]


class RecordError(ValueError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


def detect(rec: dict):
    if isinstance(rec.get("messages"), list):
        return "messages"
    if isinstance(rec.get("conversations"), list):
        return "sharegpt"
    if "instruction" in rec and ("output" in rec or "response" in rec):
        return "alpaca"
    for p, c in PC_PAIRS:
        if p in rec and c in rec:
            return "prompt_completion"
    for v in rec.values():  # any list of {"role","content"} turns, e.g. ParisNeo's "discussion" column
        if isinstance(v, list) and v and all(isinstance(m, dict) and "role" in m and "content" in m for m in v):
            return "messages_col"
    return None


def _role(r):
    role = ROLE_MAP.get(str(r).strip().lower())
    if role is None:
        raise RecordError("bad_role", repr(r))
    return role


def _text(v):
    if not isinstance(v, str):
        raise RecordError("bad_content", f"expected str, got {type(v).__name__}")
    return v


def to_messages(rec: dict, fmt: str, columns: dict | None = None) -> list[dict]:
    """fmt: messages | sharegpt | alpaca | prompt_completion | columns ({"user":col,"assistant":col,"system"?:col})."""
    if fmt == "columns":
        try:
            msgs = [{"role": "user", "content": _text(rec[columns["user"]])},
                    {"role": "assistant", "content": _text(rec[columns["assistant"]])}]
        except KeyError as e:
            raise RecordError("missing_field", f"{e} not in {sorted(rec)[:8]}")
        if columns.get("system") and rec.get(columns["system"]):
            msgs.insert(0, {"role": "system", "content": _text(rec[columns["system"]])})
        return msgs
    if fmt == "messages":
        return [{"role": _role(m.get("role")), "content": _text(m.get("content"))} for m in rec["messages"]]
    if fmt == "sharegpt":
        return [{"role": _role(m.get("from", m.get("role"))), "content": _text(m.get("value", m.get("content")))}
                for m in rec["conversations"]]
    if fmt == "alpaca":
        user = _text(rec["instruction"]).strip()
        extra = rec.get("input") or rec.get("context")  # Alpaca "input", Dolly "context"
        if extra:
            user += "\n\n" + _text(extra).strip()
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": _text(rec.get("output", rec.get("response")))}]
        if rec.get("system"):
            msgs.insert(0, {"role": "system", "content": _text(rec["system"])})
        return msgs
    if fmt == "prompt_completion":
        for p, c in PC_PAIRS:
            if p in rec and c in rec:
                return [{"role": "user", "content": _text(rec[p])}, {"role": "assistant", "content": _text(rec[c])}]
        raise RecordError("missing_field", f"no prompt/completion pair in {sorted(rec)[:8]}")
    if fmt == "messages_col":
        key = next(k for k, v in rec.items() if isinstance(v, list) and v and all(
            isinstance(m, dict) and "role" in m and "content" in m for m in v))
        msgs = [{"role": _role(m["role"]), "content": _text(m["content"])} for m in rec[key]]
        sysp = rec.get("system_prompt") or rec.get("system")
        if sysp and msgs and msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": _text(sysp)})
        return msgs
    raise RecordError("unknown_format", fmt)


def normalize(msgs: list[dict], default_system: str | None = None, replace_system: bool = False) -> list[dict]:
    msgs = [{"role": m["role"], "content": m["content"].strip()} for m in msgs]
    if not msgs:
        raise RecordError("empty", "no messages")
    if any(not m["content"] for m in msgs):
        raise RecordError("empty", "blank message content")
    body = msgs[1:] if msgs[0]["role"] == "system" else msgs
    if not body or body[-1]["role"] != "assistant":
        raise RecordError("no_assistant_last", "conversation must end with an assistant turn")
    for i, m in enumerate(body):
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            raise RecordError("bad_order", f"turn {i} is {m['role']}, expected alternating user/assistant")
    if default_system and (replace_system or msgs[0]["role"] != "system"):
        if msgs[0]["role"] == "system":
            msgs = msgs[1:]
        msgs.insert(0, {"role": "system", "content": default_system})
    return msgs
