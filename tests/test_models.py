import json

import pytest

from ftstudio import config, models
from ftstudio.train_utils import build_config
from ftstudio.workers.train import make_trainer


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_HOME", str(tmp_path))


def test_catalog_defaults_drive_training_config():
    q = build_config({"base_model": "unsloth/Qwen3.5-2B"})
    assert q["load_in_4bit"] is False and q["loader"] == "language"      # Qwen3.5: no QLoRA
    g = build_config({"base_model": "unsloth/gemma-4-E2B-it"})
    assert g["load_in_4bit"] is True and g["loader"] == "model"
    assert build_config({"base_model": "someone/unknown"})["load_in_4bit"] is True
    assert build_config({"base_model": "unsloth/Qwen3.5-2B", "load_in_4bit": True})["load_in_4bit"] is True  # explicit wins
    with pytest.raises(ValueError):
        build_config({"base_model": "x", "loader": "bogus"})


def test_user_catalog_extends_and_overrides():
    (config.home() / "models.json").write_text(json.dumps([{"id": "me/my-model", "vram_gb": 6, "load_in_4bit": False},
                                                            {"id": "unsloth/Qwen3.5-2B", "vram_gb": 99}]))
    assert models.find("me/my-model")["loader"] == "language"
    assert models.find("unsloth/Qwen3.5-2B")["vram_gb"] == 99
    (config.home() / "models.json").write_text("{broken")
    assert models.find("unsloth/Qwen3.5-2B")["vram_gb"] == 5             # bad file ignored


def test_notes_and_quant_warning():
    assert "16-bit LoRA" in models.catalog_note("unsloth/Qwen3.5-4B") and models.catalog_note("nope") is None
    assert models.quant_warning("unsloth/gemma-4-E2B-it", ["q4_k_m"]) and not models.quant_warning("unsloth/gemma-4-E2B-it", ["q8_0"])
    assert models.quant_warning("unsloth/Qwen3.5-2B", ["q4_k_m"]) is None


def test_every_builtin_is_well_formed():
    ids = [m["id"] for m in models.BUILTIN]
    assert len(ids) == len(set(ids)) and all(m["loader"] in ("language", "model") and m["vram_gb"] for m in models.BUILTIN)


def test_make_trainer_handles_both_trl_generations():
    class New:
        def __init__(self, processing_class=None, model=None):
            self.tok = processing_class

    class Old:
        def __init__(self, tokenizer=None, model=None):  # strict, like the real signature
            self.tok = tokenizer

    assert make_trainer(New, "T", model=1).tok == "T" and make_trainer(Old, "T", model=1).tok == "T"


def test_models_endpoint():
    from fastapi.testclient import TestClient
    from ftstudio.api import app
    with TestClient(app) as c:
        r = c.get("/models").json()
        assert any(m["id"] == "unsloth/Qwen3.5-2B" for m in r)
