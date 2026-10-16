"""Response-only loss masking for engines without Unsloth's train_on_responses_only (peft). Pure functions, no torch.
instruction_part / response_part are the user/assistant header strings measured by the prep template probe."""


def assistant_spans(text: str, inst: str, resp: str) -> list[tuple[int, int]]:
    """Character spans of every assistant reply, from the end of its header to the next user header (or end of text).
    The span includes the end-of-turn token, so the model learns to stop."""
    spans, i = [], 0
    while True:
        j = text.find(resp, i)
        if j < 0:
            return spans
        start = j + len(resp)
        k = text.find(inst, start)
        end = k if k >= 0 else len(text)
        spans.append((start, end))
        i = end


def mask_labels(input_ids: list[int], offsets: list[tuple[int, int]], spans) -> list[int]:
    labels = [-100] * len(input_ids)
    for t, (a, b) in enumerate(offsets):
        if b > a and any(s <= a < e for s, e in spans):
            labels[t] = input_ids[t]
    return labels


def tokenize_example(tok, messages, inst: str, resp: str, max_len: int):
    """{"input_ids","labels"} or None (no assistant tokens found, or longer than max_len: dropped, never truncated)."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    labels = mask_labels(ids, [tuple(o) for o in enc["offset_mapping"]], assistant_spans(text, inst, resp))
    if len(ids) > max_len or not any(l != -100 for l in labels):
        return None
    return {"input_ids": ids, "labels": labels}
