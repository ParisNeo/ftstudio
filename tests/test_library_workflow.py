import json
from pathlib import Path

import pytest

from ftstudio.data import formats as F
from ftstudio.data.prep import prepare
from ftstudio.evals import custom

ROW = {"system_prompt": "You are an AI assistant ... based *only* on the information provided in this conversation.",
       "discussion": [{"role": "user", "content": "How do I log JSON?"}, {"role": "assistant", "content": "Use JSONFormatter."},
                      {"role": "user", "content": "And to a file?"}, {"role": "assistant", "content": "Use FileHandler."}]}


def test_discussion_layout_is_detected_and_system_prompt_kept():
    assert F.detect(ROW) == "messages_col"
    m = F.normalize(F.to_messages(ROW, "messages_col"))
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant"]
    assert m[0]["content"].startswith("You are an AI assistant")


def test_replace_system_overrides_per_row_prompts():
    m = F.normalize(F.to_messages(ROW, "messages_col"), "NEUTRAL", replace_system=True)
    assert m[0] == {"role": "system", "content": "NEUTRAL"} and len(m) == 5
    kept = F.normalize(F.to_messages(ROW, "messages_col"), "NEUTRAL")           # default: existing system wins
    assert kept[0]["content"].startswith("You are an AI assistant")


def test_prepare_the_real_dataset_shape(tmp_path):
    rows = [{**ROW, "discussion": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(36)]
    rep = prepare({"hf_datasets": ["ParisNeo/ascii_colors_discussions"], "system_prompt": "NEUTRAL", "replace_system": True,
                   "val_fraction": 0.1}, tmp_path, hf_loader=lambda s: iter(rows))
    assert rep["kept"] == 36 and rep["val"] >= 1 and rep["formats"] == {"messages_col": 36}
    first = json.loads(open(rep["paths"]["train"]).readline())["messages"]
    assert first[0]["content"] == "NEUTRAL"


def test_python_mode_runs_code_and_explains_failures():
    ok = custom.score_detail({"match": "python", "expected": "hi"}, "```python\nprint('hi')\n```")
    assert ok == (True, "")
    assert custom.score_detail({"match": "python", "expected": "hi"}, "```python\nprint('bye')\n```")[0] is False
    ok, detail = custom.score_detail({"match": "python"}, "```python\nimport nonexistent_pkg_xyz\n```")
    assert not ok and "ModuleNotFoundError" in detail
    assert custom.score_detail({"match": "python", "timeout": 1}, "```python\nimport time; time.sleep(5)\n```")[1].startswith("exit 124")
    assert custom.score({"match": "python", "expected": ["a", "b"]}, "print('a b')")                # no fence: whole text is code


def test_starter_eval_is_satisfiable_and_catches_invented_apis():
    pytest.importorskip("ascii_colors")
    items = custom.load(str(Path(__file__).parents[1] / "examples" / "ascii_colors_eval.jsonl"))
    assert len(items) >= 8
    for it in items:
        ok, detail = custom.score_detail(it, "```python\n" + it["reference"] + "\n```")
        assert ok, (it["id"], detail)
    bad = "```python\nfrom ascii_colors import ASCIIColors\nASCIIColors.rainbow('Hello')\n```"
    assert custom.score_detail(items[0], bad)[0] is False
