import json
from pathlib import Path

import pytest

from ftstudio.data import formats as F
from ftstudio.data.prep import load_records, prepare, template_probe
from ftstudio.workers import prep as prep_worker


class StubTok:
    chat_template = "stub"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        return "".join(f"<{m['role']}>{m['content']}</s>" for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def jl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_detect_and_convert_all_formats():
    cases = {
        "alpaca": {"instruction": "Q", "input": "ctx", "output": "A"},
        "sharegpt": {"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]},
        "messages": {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
        "prompt_completion": {"question": "Q", "answer": "A"},
    }
    for fmt, rec in cases.items():
        assert F.detect(rec) == fmt
        m = F.normalize(F.to_messages(rec, fmt))
        assert [x["role"] for x in m] == ["user", "assistant"]
    assert F.to_messages(cases["alpaca"], "alpaca")[0]["content"] == "Q\n\nctx"


@pytest.mark.parametrize("msgs,reason", [
    ([{"role": "user", "content": "q"}], "no_assistant_last"),
    ([{"role": "user", "content": " "}, {"role": "assistant", "content": "a"}], "empty"),
    ([{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "bad_order"),
])
def test_normalize_rejects(msgs, reason):
    with pytest.raises(F.RecordError) as e:
        F.normalize(msgs)
    assert e.value.reason == reason


def test_bad_role_and_default_system():
    with pytest.raises(F.RecordError) as e:
        F.to_messages({"messages": [{"role": "wizard", "content": "x"}]}, "messages")
    assert e.value.reason == "bad_role"
    m = F.normalize([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "SYS")
    assert m[0] == {"role": "system", "content": "SYS"}


def test_csv_with_columns_and_load_errors(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("q,a\nhello,world\n")
    rep = prepare({"inputs": [str(f)], "columns": {"user": "q", "assistant": "a"}, "val_fraction": 0}, tmp_path)
    assert rep["kept"] == 1 and rep["formats"] == {"columns": 1}
    bad = tmp_path / "b.jsonl"
    bad.write_text('{"prompt":"p","completion":"c"}\n{not json\n')
    assert [e for _, _, e in load_records(str(bad))][1] is not None
    with pytest.raises(ValueError):
        list(load_records(str(tmp_path / "x.pdf")))


def test_prepare_drops_dedupes_decontaminates_filters(tmp_path):
    rows = [
        {"instruction": "What is 2+2?", "output": "4"},
        {"instruction": "what is  2+2?", "output": "4"},                     # duplicate after normalization
        {"instruction": "Capital of France?", "output": "Paris"},            # contaminated
        {"instruction": "long " * 50, "output": "x"},                        # too long
        {"instruction": "ok", "output": ""},                                 # empty
        {"weird": 1},                                                        # unknown format
        {"instruction": "Say hi", "output": "hi"},
    ]
    ev = jl(tmp_path / "ev.jsonl", [{"id": "a", "prompt": "capital of france?", "expected": "paris"}])
    rep = prepare({"inputs": [jl(tmp_path / "d.jsonl", rows)], "decontaminate_against": [ev],
                   "max_seq_len": 30, "val_fraction": 0}, tmp_path, StubTok())
    assert rep["read"] == 7 and rep["kept"] == 2
    assert rep["dropped"] == {"duplicate": 1, "contaminated": 1, "too_long": 1, "empty": 1, "unknown_format": 1}
    assert rep["read"] == rep["kept"] + sum(rep["dropped"].values())
    assert rep["length_unit"] == "tokens" and rep["template_probe"]["between"] == "</s><assistant>"
    assert "<user>" in (tmp_path / "preview.txt").read_text()


def test_split_is_deterministic_and_prompt_stable(tmp_path):
    rows = [{"prompt": f"question number {i}", "completion": f"a{i}"} for i in range(200)]
    src = jl(tmp_path / "d.jsonl", rows)
    a = prepare({"inputs": [src], "val_fraction": 0.1}, tmp_path)
    b = prepare({"inputs": [src], "val_fraction": 0.1}, tmp_path)
    assert a["sha256"] == b["sha256"] and 5 <= a["val"] <= 40 and a["train"] + a["val"] == 200
    val_prompts = {json.loads(l)["messages"][0]["content"] for l in open(a["paths"]["val"])}
    train_prompts = {json.loads(l)["messages"][0]["content"] for l in open(a["paths"]["train"])}
    assert not val_prompts & train_prompts


def test_tiny_dataset_gets_a_val_example(tmp_path):
    rep = prepare({"inputs": [jl(tmp_path / "d.jsonl", [{"prompt": "a", "completion": "b"}, {"prompt": "c", "completion": "d"}])],
                   "val_fraction": 0.0001}, tmp_path)
    assert rep["val"] == 1 and rep["train"] == 1


def test_max_len_requires_tokenizer(tmp_path):
    with pytest.raises(ValueError):
        prepare({"inputs": [], "max_seq_len": 10}, tmp_path)


def test_probe_flags_templates_that_mangle_content():
    class Bad(StubTok):
        def apply_chat_template(self, msgs, **kw):
            return "nothing"
    assert "error" in template_probe(Bad())


def test_worker_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(prep_worker, "load_tokenizer", lambda name: StubTok())
    run = tmp_path / "run"
    run.mkdir()
    src = jl(tmp_path / "d.jsonl", [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(30)])
    (run / "params.json").write_text(json.dumps({"inputs": [src], "tokenizer": "stub", "max_seq_len": 100}))
    prep_worker.main(str(run))
    r = json.loads((run / "result.json").read_text())
    assert r["kept"] == 30 and (run / "train.jsonl").exists() and (run / "report.json").exists()


def test_real_hf_tokenizer_path(tmp_path):
    pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(["[UNK]", "<|im_start|>", "<|im_end|>", "user", "assistant", "hi", "there", "\n"])}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")
    fast.chat_template = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    fast.save_pretrained(tmp_path / "tok")
    tok = prep_worker.load_tokenizer(str(tmp_path / "tok"))
    probe = template_probe(tok)
    assert probe["between"] == "<|im_end|>\n<|im_start|>assistant\n" and probe["prefix"] == "<|im_start|>user\n"
    rep = prepare({"inputs": [jl(tmp_path / "d.jsonl", [{"prompt": "hi", "completion": "there"}] * 1)],
                   "max_seq_len": 50, "val_fraction": 0}, tmp_path, tok)
    assert rep["kept"] == 1 and rep["length_stats"]["max"] > 0
