import json

import pytest
from fastapi.testclient import TestClient

from ftstudio import config, presets
from ftstudio.train_utils import build_config
from ftstudio.workers import KINDS


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_HOME", str(tmp_path))


def test_builtins_are_well_formed():
    ids = [p["id"] for p in presets.BUILTIN]
    assert len(ids) == len(set(ids))
    for p in presets.BUILTIN:
        assert p["kind"] in KINDS and p["name"] and p["desc"] and isinstance(p["params"], dict)


def test_train_presets_build_valid_configs():
    for p in presets.BUILTIN:
        if p["kind"] == "train":
            build_config({**p["params"], "base_model": "m"})


def test_user_presets_merge_override_and_ignore_junk():
    (config.home() / "presets.json").write_text(json.dumps([
        {"id": "mine", "kind": "train", "name": "Mine", "params": {"epochs": 2}},
        {"id": "train-smoke", "kind": "train", "name": "Override", "params": {}},
        {"id": "bad"}, "junk"]))
    by = {p["id"]: p for p in presets.all_presets()}
    assert by["mine"]["desc"] == "" and by["train-smoke"]["name"] == "Override" and "bad" not in by
    (config.home() / "presets.json").write_text("{not json")
    assert len(presets.all_presets()) == len(presets.BUILTIN)


def test_presets_endpoint_and_start_page():
    from ftstudio.api import app
    with TestClient(app) as c:
        assert len(c.get("/presets").json()) == len(presets.BUILTIN)
        assert "Start here" in c.get("/").text
