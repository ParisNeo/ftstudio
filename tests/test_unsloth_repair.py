import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from ftstudio import doctor
from ftstudio.workers import train, train_peft

# real requires_dist lines of unsloth 2026.10.5 (read from PyPI), trimmed
META = {"info": {"requires_dist": [
    "unsloth_zoo>=2026.10.5", "torch<2.15.0,>=2.4.0", "xformers>=0.0.27.post2; sys_platform == \"win32\"",
    "bitsandbytes!=0.46.0,!=0.48.0,>=0.45.5", "datasets!=4.0.*,!=4.1.0,!=4.4.*,!=4.5.0,<5.0.0,>=3.4.1", "accelerate>=0.34.1",
    "peft!=0.11.0,>=0.18.0", "transformers!=4.53.0,!=5.0.0,!=5.1.0,<=5.17.0,>=4.52.4", "trl!=0.19.0,<=1.15.0,>=0.18.2", "huggingface_hub>=0.34.0",
    "transformers>=9; extra == \"huggingface\"", "triton>=3.0.0; \"linux\" in sys_platform and extra == \"triton\""]}}


def installer():
    spec = importlib.util.spec_from_file_location("ftstudio_install2", Path(__file__).parents[1] / "install.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bounds_come_from_unsloths_own_metadata():
    m = installer()
    b = m.unsloth_bounds(META)
    assert [x.split("<")[0].split(">")[0].split("!")[0] for x in b] == ["accelerate", "bitsandbytes", "datasets", "peft", "transformers", "trl"]
    assert "transformers!=4.53.0,!=5.0.0,!=5.1.0,<=5.17.0,>=4.52.4" in b and "datasets!=4.0.*,!=4.1.0,!=4.4.*,!=4.5.0,<5.0.0,>=3.4.1" in b
    assert not any("extra" in x or "triton" in x or "xformers" in x for x in b)          # extras and unrelated packages are ignored


def test_constraints_pin_torch_and_bound_the_libraries():
    m = installer()
    txt = m.constraints_text(["torch==2.14.1+cu132", "numpy==2.0", "torchvision==0.29.1+cu132"], ["transformers<=5.17.0"])
    assert txt.splitlines() == ["torch==2.14.1+cu132", "torchvision==0.29.1+cu132", "transformers<=5.17.0"]


def test_stale_unsloth_is_detected():
    m = installer()
    assert m.is_modern("2026.10.5") and not m.is_modern("2025.7.2") and not m.is_modern(None)     # the version that broke your install


def test_bounds_fetch_failure_is_not_fatal(monkeypatch, capsys):
    m = installer()
    monkeypatch.setattr(m.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert m.unsloth_bounds() == [] and "could not read" in capsys.readouterr().out


def make_job(tmp_path, backend):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "report.json").write_text(json.dumps({"val": 0, "sha256": "s", "tokenizer": None,
                                               "template_probe": {"instruction_part": "<u>", "response_part": "<a>"}}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "params.json").write_text(json.dumps({"dataset": {"dir": str(d)}, "base_model": "org/model", "backend": backend}))
    return run


@pytest.fixture()
def broken_unsloth(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))     # importable, but has no FastLanguageModel
    monkeypatch.delitem(sys.modules, "unsloth.chat_templates", raising=False)
    monkeypatch.setattr(train, "require_accelerator", lambda **k: {"accelerator": "cuda", "vendor": "nvidia", "status": "ok"})
    monkeypatch.setattr(train, "choose_backend", lambda req, rep, **k: "unsloth")


def test_auto_falls_back_to_peft_when_unsloth_cannot_import(tmp_path, monkeypatch, broken_unsloth, capsys):
    calls = []
    monkeypatch.setattr(train_peft, "run_peft", lambda *a, **k: calls.append(a))
    train.main(str(make_job(tmp_path, "auto")))
    assert len(calls) == 1 and "Falling back to the generic peft engine" in capsys.readouterr().out


def test_explicit_unsloth_still_fails_loudly(tmp_path, monkeypatch, broken_unsloth):
    monkeypatch.setattr(train_peft, "run_peft", lambda *a, **k: pytest.fail("must not fall back when unsloth was requested"))
    with pytest.raises(ImportError):
        train.main(str(make_job(tmp_path, "unsloth")))


def deep_report(backends):
    return lambda **k: {"status": "ok", "message": "ok", "fix": "", "accelerator": "cuda", "backends": list(backends), "vendor": "nvidia"}


def test_doctor_deep_demotes_broken_unsloth_to_a_warning_when_peft_works(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "unsloth", None)                                # `import unsloth` now raises ImportError
    monkeypatch.setattr(doctor, "report", deep_report(["unsloth", "peft"]))
    assert doctor.main(["--deep", "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["status"] == "ok" and r["backends"] == ["peft"] and r["unsloth"].startswith("BROKEN") and "peft engine" in r["note"]
    assert isinstance(r["versions"], dict)


def test_doctor_deep_fails_when_unsloth_was_the_only_engine(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "unsloth", None)
    monkeypatch.setattr(doctor, "report", deep_report(["unsloth"]))
    assert doctor.main(["--deep", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "unsloth_broken"


def run_real_flow(tmp_path, monkeypatch, hw, args, unsloth_ver="2026.10.5", doctor_reports=None):
    """Runs install.main() NON-dry with every external command mocked, so the post-install code paths execute."""
    m = installer()
    venv = tmp_path / "v"
    py = m.venv_python(venv)
    py.parent.mkdir(parents=True)
    py.write_text("")                                                           # venv "exists"
    cmds = []
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "detect", lambda: hw)
    monkeypatch.setattr(m, "run", lambda cmd, dry, cwd=None, check=True: cmds.append([str(c) for c in cmd]))
    monkeypatch.setattr(m, "unsloth_bounds", lambda meta=None: ["transformers<=5.17.0"])
    monkeypatch.setattr(m, "unsloth_version", lambda py: unsloth_ver)
    monkeypatch.setattr(m, "torch_state", lambda *a, **k: (True, "2.14.1+cu132 accelerator=cuda"))
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="torch==2.14.1+cu132\ntorchvision==0.29.1+cu132\nnumpy==2\n", returncode=0))
    reports = iter(doctor_reports or [{"status": "ok", "message": "fine", "backends": ["unsloth", "peft"]}])
    monkeypatch.setattr(m, "doctor_json", lambda py: next(reports))
    m.main(["--venv", str(venv), *args])
    return cmds, venv


NV = {"vendor": "nvidia", "name": "RTX 3060", "driver": "610.88", "driver_cuda": "13.0", "os": "Windows", "arch": "amd64"}


def test_real_flow_core_only_does_not_crash_after_install(tmp_path, monkeypatch):      # the KeyError this smoke test found
    cmds, _ = run_real_flow(tmp_path, monkeypatch, {"vendor": None, "name": None, "os": "Linux", "arch": "x86_64"}, [],
                            doctor_reports=[{"status": "no_gpu", "message": "no gpu"}])
    assert any("-e" in c and "." in c for c in cmds) and not any("torch" in " ".join(c) for c in cmds)
    assert (tmp_path / "run.sh").exists() and (tmp_path / "run.bat").exists()


def test_real_flow_nvidia_pins_torch_and_bounds_and_does_not_repair_a_healthy_install(tmp_path, monkeypatch):
    cmds, venv = run_real_flow(tmp_path, monkeypatch, NV, [])
    cons = (venv / "torch-constraints.txt").read_text().splitlines()
    assert cons == ["torch==2.14.1+cu132", "torchvision==0.29.1+cu132", "transformers<=5.17.0"]
    assert any(".[train]" in c and "-c" in c for c in cmds) and not any("unsloth>=2026.1" in c for c in cmds)


def test_real_flow_repairs_a_stale_unsloth_and_a_broken_import(tmp_path, monkeypatch):
    cmds, _ = run_real_flow(tmp_path, monkeypatch, NV, [], unsloth_ver="2025.7.2")        # your install: pip fell back to 2025.7.2
    assert any("unsloth>=2026.1" in c and "-c" in c for c in cmds)
    cmds, _ = run_real_flow(tmp_path / "b", monkeypatch, NV, [], doctor_reports=[
        {"status": "ok", "message": "m", "unsloth": "BROKEN: ImportError(...)", "backends": ["peft"]},
        {"status": "ok", "message": "m", "backends": ["unsloth", "peft"]}])
    assert sum("unsloth>=2026.1" in c for c in cmds) == 1
    cmds, _ = run_real_flow(tmp_path / "c", monkeypatch, NV, ["--repair"])                 # explicit --repair, healthy install
    assert any("unsloth>=2026.1" in c for c in cmds) and not any("--force-reinstall" in c for c in cmds)   # no big torch download
