import json

import pytest

from ftstudio import library
from ftstudio.data import synth
from ftstudio.data.prep import prepare


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_HOME", str(tmp_path))


def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir(exist_ok=True)
    (d / "guide.md").write_text(("Alpha " * 35) + "\n\n" + ("Bravo " * 35))
    return str(d)


GOOD = {"question": "How do I print green text with the library?", "answer": "Use green.\n```python\nprint('ok-green')\n```"}
BROKEN = {"question": "How do I print rainbow text with it?", "answer": "Call rainbow.\n```python\nundefined_function()\n```"}
DUP = {"question": "how do I print  green text with the library?", "answer": "A long enough duplicate answer text."}
SELFREF = {"question": "What does the excerpt say about colors?", "answer": "A long enough answer about colors here."}
WRONG = {"question": "Why is this wrong fact true here?", "answer": "WRONG claim that is long enough to pass the length check."}
FINE = {"question": "What is the purpose of the library overall?", "answer": "It prints colored text and handles logging nicely."}


def make_chat(chunk_pairs):
    state = {"gen": 0, "calls": []}

    def chat(messages, temperature=0.3, max_tokens=1000):
        user = messages[-1]["content"]
        state["calls"].append(user[:30])
        if "strict reviewer" in user:
            return "FAIL: invented claim" if "WRONG" in user else "PASS"
        if "failed when it was run" in user:
            return '<think>fixing</think>{"answer": "Use print.\\n```python\\nprint(\'fixed\')\\n```"}'
        pairs = chunk_pairs[state["gen"]]
        state["gen"] += 1
        return "Here you go:\n" + json.dumps(pairs)

    chat.state = state
    return chat


def test_split_and_parse_helpers():
    assert len(synth.split_text(("Alpha " * 35) + "\n\n" + ("Bravo " * 35), 250, 50)) == 2
    with pytest.raises(ValueError):
        synth.split_text("x", 100, 100)
    assert synth.parse_pairs('<think>x</think>```json\n[{"question": "q?", "answer": "a"}]\n```') == [{"question": "q?", "answer": "a"}]
    assert synth.parse_pairs('{"pairs": [{"question": "q", "answer": "a"}, {"question": "", "answer": "x"}]}')[0]["answer"] == "a"
    assert synth.parse_pairs("not json at all") == [] and synth.parse_answer('{"answer": " fixed "}') == "fixed"


def test_full_pipeline_counts_repairs_and_metadata(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    chat = make_chat([[GOOD, BROKEN, DUP, SELFREF], [WRONG, FINE]])
    r = synth.run({"sources": [docs(tmp_path)], "library": "lib1", "chunk_size": 250, "chunk_overlap": 50, "per_chunk": 4,
                   "system_prompt": "You are an expert."}, chat, out, model="m")
    assert (r["chunks"], r["generated"], r["kept"], r["repaired"]) == (2, 6, 3, 1)
    assert r["rejected"] == {"duplicate": 1, "refers_to_excerpt": 1, "judge_rejected": 1}
    rows = library.read_all("lib1")
    assert len(rows) == 3 and rows[0]["messages"][0] == {"role": "system", "content": "You are an expert."}
    assert rows[1]["meta"]["code"] == "repaired" and "fixed" in rows[1]["messages"][-1]["content"]
    assert rows[0]["meta"] == {"source": "guide.md", "chunk": 0, "model": "m", "judge": "PASS", "code": "ok"}
    rej = [json.loads(l) for l in open(out / "rejects.jsonl")]
    assert {x["reason"] for x in rej} == {"duplicate", "refers_to_excerpt", "judge_rejected"}


def test_dry_run_never_touches_library_and_modes(tmp_path):
    base = {"sources": [docs(tmp_path)], "library": "lib2", "chunk_size": 250, "chunk_overlap": 50, "per_chunk": 4}
    out = tmp_path / "r1"
    out.mkdir()
    synth.run({**base, "dry_run": True}, make_chat([[GOOD], [FINE]]), out)
    assert library.read_all("lib2") == [] and len(open(out / "preview.jsonl").read().splitlines()) == 2
    out2 = tmp_path / "r2"
    out2.mkdir()
    synth.run(base, make_chat([[GOOD], [FINE]]), out2)
    assert len(library.read_all("lib2")) == 2
    out3 = tmp_path / "r3"
    out3.mkdir()
    r = synth.run(base, make_chat([[GOOD, FINE], [FINE]]), out3)          # append: old questions are duplicates
    assert r["kept"] == 0 and r["rejected"] == {"duplicate": 3} and len(library.read_all("lib2")) == 2
    with pytest.raises(ValueError, match="already has entries"):
        synth.run({**base, "mode": "create"}, make_chat([[GOOD], [FINE]]), out3)
    r = synth.run({**base, "mode": "create", "overwrite": True}, make_chat([[WRONG], [FINE]]), out3)
    assert len(library.read_all("lib2")) == 1 and r["kept"] == 1


def test_must_import_unparseable_empty_and_contamination(tmp_path):
    ev = tmp_path / "ev.jsonl"
    ev.write_text(json.dumps({"id": "a", "prompt": FINE["question"], "expected": "x"}) + "\n")
    out = tmp_path / "run"
    out.mkdir()
    chat = make_chat([[{"question": "How do I use sys here properly?", "answer": "Do it.\n```python\nimport os\nprint(1)\n```"}, FINE], "[]"])
    r = synth.run({"sources": [docs(tmp_path)], "library": "lib3", "chunk_size": 250, "chunk_overlap": 50, "per_chunk": 3,
                   "must_import": "sys", "decontaminate_against": [str(ev)]}, chat, out)
    assert r["rejected"] == {"no_library_import": 1, "contaminated": 1} and r["kept"] == 0
    out2 = tmp_path / "run2"
    out2.mkdir()
    bad = lambda m, **k: "I refuse" if "Documentation excerpt" in m[-1]["content"] else "PASS"
    r = synth.run({"sources": [docs(tmp_path)], "library": "lib4", "chunk_size": 250, "chunk_overlap": 50}, bad, out2)
    assert r["rejected"] == {"unparseable_output": 2}


def test_prompt_must_contain_chunk_and_custom_prompt_is_used(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(ValueError, match="chunk"):
        synth.run({"sources": [docs(tmp_path)], "library": "l", "prompt": "no placeholder"}, make_chat([]), out)
    seen = []

    def chat(m, **k):
        seen.append(m[-1]["content"])
        return "[]"
    synth.run({"sources": [docs(tmp_path)], "library": "l", "chunk_size": 250, "chunk_overlap": 50,
               "prompt": ["Be terse about {source}:", "{chunk}", "make {n}"], "per_chunk": 2}, chat, out)
    assert seen[0].startswith("Be terse about guide.md:\nAlpha") and seen[0].endswith("make 2")


def test_library_feeds_prepare_and_api(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    synth.run({"sources": [docs(tmp_path)], "library": "lib5", "chunk_size": 250, "chunk_overlap": 50}, make_chat([[GOOD], [FINE]]), out)
    rep = prepare({"libraries": ["lib5"], "val_fraction": 0}, tmp_path)
    assert rep["kept"] == 2 and rep["sources"] == {"library:lib5": 2} and rep["formats"] == {"messages": 2}
    from fastapi.testclient import TestClient
    from ftstudio.api import app
    with TestClient(app) as c:
        assert c.get("/library").json() == [{"name": "lib5", "entries": 2, "bytes": library.path("lib5").stat().st_size}]
        assert c.get("/library/lib5?limit=1").json()["total"] == 2
        assert c.get("/library/..bad").status_code in (400, 404)
        assert c.delete("/library/lib5/0").json() == {"deleted": 0} and c.delete("/library/lib5/5").status_code == 404
        assert c.get("/library/lib5").json()["total"] == 1
