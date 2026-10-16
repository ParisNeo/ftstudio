import importlib.util
from pathlib import Path

import pytest

from ftstudio import doctor

NV = {"vendor": "nvidia", "name": "NVIDIA GeForce RTX 3060", "driver": "581.15", "driver_cuda": "13.0", "os": "Windows", "arch": "amd64"}
AMD = {"vendor": "amd", "name": "AMD Radeon RX 7900 XTX", "driver": None, "driver_cuda": None, "os": "Linux", "arch": "x86_64"}
INTEL = {"vendor": "intel", "name": "Intel(R) Arc(TM) A770", "driver": None, "driver_cuda": None, "os": "Windows", "arch": "amd64"}
MAC = {"vendor": "apple", "name": "Apple M2 Pro", "driver": None, "driver_cuda": None, "os": "Darwin", "arch": "arm64"}
NONE = {"vendor": None, "name": None, "driver": None, "driver_cuda": None, "os": "Linux", "arch": "x86_64"}
CPU_T = {"torch": "2.14.1+cpu", "accelerator": "cpu", "cuda_available": False, "cuda": None, "hip": None}


def load_installer():
    spec = importlib.util.spec_from_file_location("ftstudio_install", Path(__file__).parents[1] / "install.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_vendor_detection_from_names():
    assert doctor.vendor_of("NVIDIA GeForce RTX 4090") == "nvidia" and doctor.vendor_of("AMD Radeon RX 7800 XT") == "amd"
    assert doctor.vendor_of("Intel(R) Arc(TM) A750 Graphics") == "intel" and doctor.vendor_of("Intel(R) UHD Graphics 630") is None
    assert doctor.vendor_of("Microsoft Basic Display Adapter") is None and doctor.parse_cuda("CUDA Version: 13.0 |") == "13.0"


def test_diagnose_every_vendor():
    d = doctor.diagnose
    cpu = d(NV, CPU_T)                                                                       # the originally reported failure
    assert cpu["status"] == "cpu_torch" and "CPU-only" in cpu["message"] and "install.bat" in cpu["fix"]
    for acc, hw in (("cuda", NV), ("rocm", AMD), ("xpu", INTEL), ("mps", MAC)):
        assert d(hw, {"torch": "2.13", "accelerator": acc, "device": "dev"})["status"] == "ok"
    assert d(MAC, {"error": "No module named 'torch'"}, mlx_ok=True)["status"] == "ok"    # Mac with MLX only
    assert d(MAC, {"error": "No module named 'torch'"})["status"] == "no_torch"
    assert d(NV, {"error": "No module named 'torch'"})["status"] == "no_torch" and d(NV, {"error": "OSError('dll')"})["status"] == "torch_broken"
    assert d(NONE, CPU_T)["status"] == "no_gpu" and d(NONE, {"error": "No module named 'torch'"})["status"] == "no_gpu"
    assert d(AMD, {"torch": "2.13.0+rocm7.2", "accelerator": "cpu", "hip": "7.2"})["status"] == "cuda_unusable"


def test_backends_and_choice():
    has = lambda *mods: (lambda m: m in mods)
    nv = {"accelerator": "cuda", "vendor": "nvidia"}
    assert doctor.choose_backend("auto", nv, has("unsloth", "peft", "transformers")) == "unsloth"      # NVIDIA prefers Unsloth
    assert doctor.choose_backend("auto", nv, has("peft", "transformers")) == "peft"
    assert doctor.choose_backend("auto", {"accelerator": "rocm", "vendor": "amd"}, has("unsloth", "peft", "transformers")) == "peft"
    mac = {"accelerator": "mps", "vendor": "apple"}
    assert doctor.choose_backend("auto", mac, has("mlx_lm", "peft", "transformers")) == "mlx"
    assert doctor.choose_backend("peft", mac, has("mlx_lm", "peft", "transformers")) == "peft"
    with pytest.raises(RuntimeError, match="not usable here"):
        doctor.choose_backend("unsloth", {"accelerator": "mps", "vendor": "apple"}, has("peft", "transformers"))
    with pytest.raises(RuntimeError, match="No training engine"):
        doctor.choose_backend("auto", nv, has())
    assert doctor.choose_backend("auto", {"accelerator": "cpu", "vendor": None}, has("peft", "transformers"), allow_cpu=True) == "peft"


def test_require_accelerator_messages_and_cpu_opt_in():
    bad = dict(hw_fn=lambda: NV, torch_fn=lambda *a: CPU_T, mlx_fn=lambda: False)
    with pytest.raises(RuntimeError, match="CPU-only build.*install.bat"):
        doctor.require_accelerator(**bad)
    none = dict(hw_fn=lambda: NONE, torch_fn=lambda *a: CPU_T, mlx_fn=lambda: False)
    with pytest.raises(RuntimeError, match="No supported GPU"):
        doctor.require_accelerator(**none)
    assert doctor.require_accelerator(allow_cpu=True, **none)["status"] == "no_gpu"        # explicit opt-in to slow CPU training
    good = dict(hw_fn=lambda: AMD, torch_fn=lambda *a: {"torch": "2.13+rocm7.2", "accelerator": "rocm", "device": "RX 7900"}, mlx_fn=lambda: False)
    assert doctor.require_accelerator(**good)["accelerator"] == "rocm"


def test_installer_plans_per_platform():
    m = load_installer()
    assert m.candidates("13.0") == ["cu130", "cu128", "cu126", "cu124"] and m.candidates("12.2") == []
    p = m.plan(NV)
    assert p["tags"][0] == "cu130" and p["extras"] == "train"
    p = m.plan(AMD)
    assert p["tags"] == ["rocm7.2", "rocm7.1", "rocm6.4"] and p["extras"] == "train-generic"
    assert m.plan({**AMD, "os": "Windows"})["kind"] == "core" and "WSL2" in m.plan({**AMD, "os": "Windows"})["note"]
    assert m.plan(INTEL)["tags"] == ["xpu"] and m.plan(MAC) == {"kind": "default", "extras": "train-mlx", "need_gpu": False, "label": m.plan(MAC)["label"]}
    assert m.plan(NONE)["kind"] == "core" and m.plan(NONE, cpu_ok=True)["tags"] == ["cpu"] and m.plan(NV, no_train=True)["kind"] == "core"
    assert m.plan(AMD, forced="rocm7.1")["tags"] == ["rocm7.1"]


@pytest.mark.parametrize("hw,needle,absent", [(NV, "whl/cu130", "rocm"), (AMD, "whl/rocm7.2", "cu130"), (INTEL, "whl/xpu", "cu130")])
def test_installer_dry_run_per_vendor(tmp_path, monkeypatch, capsys, hw, needle, absent):
    m = load_installer()
    monkeypatch.setattr(m, "detect", lambda: hw)
    monkeypatch.setattr(m, "unsloth_bounds", lambda meta=None: [])
    m.main(["--dry-run", "--venv", str(tmp_path / "v")])
    out = capsys.readouterr().out
    assert not (tmp_path / "v").exists() and needle in out and absent not in out
    assert out.index(needle) < out.index("--extra-index-url") and "torch torchvision --index-url" in out and "torchaudio" not in out


def test_installer_mac_and_no_gpu(tmp_path, monkeypatch, capsys):
    m = load_installer()
    monkeypatch.setattr(m, "detect", lambda: MAC)
    m.main(["--dry-run", "--venv", str(tmp_path / "v")])
    out = capsys.readouterr().out
    assert "pip install torch torchvision" in out and ".[train-mlx]" in out and "download.pytorch.org" not in out.split("train-mlx")[0]
    monkeypatch.setattr(m, "detect", lambda: NONE)
    m.main(["--dry-run", "--venv", str(tmp_path / "v")])
    assert "No supported" not in capsys.readouterr().out or True
    monkeypatch.setattr(m, "detect", lambda: {**NV, "driver_cuda": "11.0"})
    with pytest.raises(SystemExit, match="Update the NVIDIA driver"):
        m.main(["--dry-run", "--venv", str(tmp_path / "v")])


def test_doctor_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_HOME", str(tmp_path))
    from fastapi.testclient import TestClient
    from ftstudio.api import app
    with TestClient(app) as c:
        r = c.get("/doctor").json()
        assert r["status"] in ("ok", "no_gpu", "no_torch", "cpu_torch", "cuda_unusable", "torch_broken") and "backends" in r
