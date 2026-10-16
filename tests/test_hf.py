import json

import pytest

from ftstudio import known_datasets
from ftstudio.data import hf
from ftstudio.data.prep import prepare


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_HOME", str(tmp_path))


def test_parse_spec():
    s = hf.parse_spec("HuggingFaceTB/smoltalk:all@train#5000")
    assert (s["repo"], s["config"], s["split"], s["limit"]) == ("HuggingFaceTB/smoltalk", "all", "train", 5000)
    d = hf.parse_spec("tatsu-lab/alpaca")
    assert d["split"] == "train" and d["limit"] is None and d["config"] is None
    assert hf.parse_spec("x/y@train[:100]")["split"] == "train[:100]"
    assert hf.parse_spec({"repo": "a/b", "limit": 3, "columns": {"user": "q", "assistant": "a"}})["columns"]["user"] == "q"
    assert hf.label(hf.parse_spec("openai/gsm8k:main@test")) == "openai/gsm8k:main@test"
    for bad in ("/etc/passwd", "../x", "a/b/c", "x y", "a/b#0", "", {"repo": "../../x"}):
        with pytest.raises(ValueError):
            hf.parse_spec(bad)


DOLLY = [{"instruction": "Summarize", "context": "Long text here", "response": "Short.", "category": "x"},
         {"instruction": "Say hi", "context": "", "response": "Hi", "category": "y"}]
TOME = [{"conversations": [{"from": "human", "value": "q1"}, {"from": "gpt", "value": "a1"}]}]
GSM = [{"question": "1+1?", "answer": "2"}, {"question": "2+2?", "answer": "4"}, {"question": "3+3?", "answer": "6"}]
FAKE = {"d/dolly": DOLLY, "m/tome": TOME, "o/gsm8k": GSM}


def fake_loader(s):
    return iter(FAKE[s["repo"]])


def test_prepare_from_hf_sources_with_mixed_formats(tmp_path):
    local = tmp_path / "mine.jsonl"
    local.write_text(json.dumps({"prompt": "own q", "completion": "own a"}) + "\n")
    rep = prepare({"inputs": [str(local)], "hf_datasets": ["d/dolly", "m/tome", "o/gsm8k#2"], "val_fraction": 0},
                  tmp_path, hf_loader=fake_loader)
    assert rep["read"] == 1 + 2 + 1 + 2 and rep["kept"] == rep["read"]                       # limit=2 respected
    assert rep["sources"] == {"mine.jsonl": 1, "d/dolly@train": 2, "m/tome@train": 1, "o/gsm8k@train": 2}
    rows = [json.loads(l)["messages"][0]["content"] for l in open(rep["paths"]["train"])]
    assert "Summarize\n\nLong text here" in rows                                            # Dolly context merged


def test_per_source_columns_and_errors(tmp_path):
    odd = [{"q": "why?", "a": "because"}]
    rep = prepare({"hf_datasets": [{"repo": "x/odd", "columns": {"user": "q", "assistant": "a"}}], "val_fraction": 0},
                  tmp_path, hf_loader=lambda s: iter(odd))
    assert rep["kept"] == 1
    with pytest.raises(ValueError):
        prepare({}, tmp_path)
    with pytest.raises(ValueError):
        prepare({"hf_datasets": ["../etc"]}, tmp_path)

    def boom(s):
        raise OSError("401 Client Error: gated repo")
    with pytest.raises(RuntimeError, match="could not load Hugging Face dataset"):
        prepare({"hf_datasets": ["g/gated"]}, tmp_path, hf_loader=boom)


def test_catalog_specs_all_parse_and_endpoint():
    for d in known_datasets.BUILTIN:
        hf.parse_spec(d["spec"])
    from fastapi.testclient import TestClient
    from ftstudio.api import app
    with TestClient(app) as c:
        assert any(d["spec"].startswith("yahma/") for d in c.get("/datasets").json())
