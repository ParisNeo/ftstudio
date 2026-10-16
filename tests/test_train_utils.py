import json

import pytest

from ftstudio import config, db
from ftstudio.train_utils import anchors, build_config, queue_evals, resolve_adapter, resolve_dataset, tokenizer_warning
from ftstudio.workers.export import quant_of
from ftstudio.workers.train import sft_config


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_HOME", str(tmp_path))
    db.init()


def fake_job(kind, status="succeeded"):
    jid = db.create_job(kind, {})
    db.claim_next()
    db.finish(jid, status)
    return jid


def test_build_config():
    c = build_config({"base_model": "m", "lora_r": 32})
    assert c["lora_alpha"] == 32 and c["epochs"] == 1 and c["grad_accum"] == 8
    for bad in ({}, {"base_model": "m", "lora_r": 0}, {"base_model": "m", "learning_rate": -1}):
        with pytest.raises(ValueError):
            build_config(bad)


def test_resolve_dataset_requires_succeeded_prep():
    jid = fake_job("prep")
    d = config.runs_dir() / jid
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({"val": 1}))
    assert resolve_dataset({"job": jid})["train"] == d / "train.jsonl"
    with pytest.raises(ValueError):
        resolve_dataset({"job": fake_job("eval")})
    with pytest.raises(ValueError):
        resolve_dataset({"job": fake_job("prep", "failed")})
    with pytest.raises(ValueError):
        resolve_dataset({})


def test_anchors():
    ok = {"template_probe": {"instruction_part": "<u>", "response_part": "<a>"}}
    assert anchors(ok) == ("<u>", "<a>")
    for bad in ({}, {"template_probe": {"error": "x"}}, {"template_probe": {"instruction_part": "<u>", "response_part": ""}}):
        with pytest.raises(ValueError):
            anchors(bad)


def test_tokenizer_warning():
    assert tokenizer_warning({"tokenizer": "meta-llama/Llama-3.1-8B"}, "unsloth/Llama-3.1-8B-bnb-4bit") is None
    assert "WARNING" in tokenizer_warning({"tokenizer": "Qwen/Qwen2.5-7B"}, "unsloth/Llama-3.1-8B")
    assert tokenizer_warning({}, "x") is None


def test_sft_config_survives_renamed_args():
    class New:
        def __init__(self, max_length=None, evaluation_strategy=None):
            self.v = (max_length, evaluation_strategy)

    assert sft_config(New, max_seq_length=7, eval_strategy="steps").v == (7, "steps")


def test_export_helpers():
    from pathlib import Path
    assert quant_of(Path("m.Q4_K_M.gguf"), ["q8_0", "q4_k_m"]) == "q4_k_m"
    jid = fake_job("train")
    assert resolve_adapter({"train_job": jid}) == config.runs_dir() / jid / "adapter"
    with pytest.raises(ValueError):
        resolve_adapter({"train_job": fake_job("prep")})
    ids = queue_evals([{"path": "/m/a.gguf", "quant": "q4_k_m"}], {"custom": {"file": "t.jsonl"}}, "run1")
    j = db.get_job(ids[0])
    assert j["kind"] == "eval" and j["params"]["name"] == "run1-q4_k_m" and j["status"] == "queued"
