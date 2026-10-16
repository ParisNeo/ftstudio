#!/usr/bin/env python3
"""ftstudio installer for Windows, Linux and macOS. Creates .venv, detects your hardware, installs the PyTorch build that matches it
(plain `pip install torch` gives the CPU-only build on Windows), installs the training engine for that hardware, verifies it, and
writes run.bat / run.sh.
  NVIDIA  -> CUDA PyTorch (newest tag your driver supports) + Unsloth          AMD (Linux) -> ROCm PyTorch + PEFT
  Intel Arc -> XPU PyTorch + PEFT                                              Apple Silicon -> MLX (mlx-lm) + PyTorch MPS
  no GPU  -> core only (prep, synth, eval), or CPU training with --cpu
  python install.py [--venv .venv] [--tag cu128|rocm7.2|xpu|cpu] [--repair] [--reinstall-torch] [--no-train] [--cpu] [--dry-run]"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX = "https://download.pytorch.org/whl/{}"
CUDA_TAGS = ["cu132", "cu130", "cu128", "cu126", "cu124"]   # newest first; each is tried until PyTorch really sees the GPU
ROCM_TAGS = ["rocm7.2", "rocm7.1", "rocm6.4"]               # Linux only
XPU_TAGS = ["xpu"]


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def tag_version(tag: str):
    n = int(tag[2:])
    return (n // 10, n % 10)


def candidates(driver_cuda, forced=None) -> list[str]:
    """CUDA wheel tags to try, newest first, never newer than what the driver supports."""
    if forced:
        return [forced]
    if not driver_cuda:
        return list(CUDA_TAGS)
    major, minor = map(int, driver_cuda.split("."))
    return [t for t in CUDA_TAGS if tag_version(t) <= (major, minor)]


def plan(hw: dict, forced=None, cpu_ok=False, no_train=False) -> dict:
    """What to install for this hardware. kind: core | torch (index tags to try) | default (plain PyPI torch, Apple)."""
    v, name = hw.get("vendor"), hw.get("name")
    if no_train:
        return {"kind": "core", "label": "core only (prep, synth, eval)", "note": ""}
    if v == "nvidia":
        return {"kind": "torch", "tags": candidates(hw.get("driver_cuda"), forced), "extras": "train", "need_gpu": True,
                "label": f"NVIDIA {name}, driver supports CUDA {hw.get('driver_cuda')}: CUDA PyTorch + Unsloth"}
    if v == "amd":
        if hw.get("os") != "Linux":
            return {"kind": "core", "label": f"AMD {name}: core only",
                    "note": "PyTorch's ROCm builds are Linux-only. On Windows use WSL2 (Ubuntu) and run this installer there, or follow Unsloth's AMD guide. "
                            "For inference, a llama.cpp Vulkan or HIP build uses the GPU."}
        return {"kind": "torch", "tags": [forced] if forced else ROCM_TAGS, "extras": "train-generic", "need_gpu": True,
                "label": f"AMD {name}: ROCm PyTorch + PEFT"}
    if v == "intel":
        return {"kind": "torch", "tags": [forced] if forced else XPU_TAGS, "extras": "train-generic", "need_gpu": True,
                "label": f"Intel {name}: XPU PyTorch + PEFT"}
    if v == "apple":
        return {"kind": "default", "extras": "train-mlx", "need_gpu": False, "label": f"Apple Silicon {name}: MLX + PyTorch (MPS)"}
    if cpu_ok:
        return {"kind": "torch", "tags": [forced or "cpu"], "extras": "train-generic", "need_gpu": False, "label": "CPU only (training will be very slow)"}
    return {"kind": "core", "label": "no supported GPU found: core only",
            "note": "Training needs an NVIDIA, AMD (Linux), Intel Arc or Apple Silicon GPU. Add --cpu to install CPU training anyway (very slow)."}


def detect():
    sys.path.insert(0, str(ROOT))
    from ftstudio.doctor import detect_hardware
    return detect_hardware()


LOG = ROOT / "install.log"
BOUND_NAMES = ("transformers", "datasets", "trl", "peft", "accelerate", "bitsandbytes")


def run(cmd, dry, cwd=None, check=True):
    """Runs a command, echoes it, and tees all output into install.log (paste that file when something fails)."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    with open(LOG, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
    if check and rc:
        raise subprocess.CalledProcessError(rc, cmd)
    return subprocess.CompletedProcess(cmd, rc, "", "")


def unsloth_bounds(meta=None) -> list[str]:
    """The version ranges Unsloth itself declares for transformers, datasets, trl, peft... read live from PyPI.
    Constraining pip to them stops it from picking, say, a transformers newer than Unsloth allows and then quietly
    falling back to a years-old Unsloth that cannot import with the new TRL."""
    try:
        meta = meta or json.load(urllib.request.urlopen("https://pypi.org/pypi/unsloth/json", timeout=20))
    except Exception as e:
        print(f"NOTE: could not read Unsloth's version bounds from PyPI ({e!r}); continuing without them.")
        return []
    out = {}
    for line in meta["info"].get("requires_dist") or []:
        if "extra ==" in line or ";" in line:
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)$", line.strip())
        if m and m.group(1).lower().replace("_", "-") in BOUND_NAMES and m.group(2):
            out[m.group(1).lower()] = m.group(1) + m.group(2).replace(" ", "")
    return sorted(out.values())


def constraints_text(freeze_lines, bounds) -> str:
    pins = [l for l in freeze_lines if re.match(r"(torch|torchvision)==", l)]
    return "\n".join(pins + list(bounds)) + "\n"


def unsloth_version(py):
    r = subprocess.run([str(py), "-c", "import importlib.metadata as m;print(m.version('unsloth'))"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def is_modern(version) -> bool:
    return bool(version) and int(version.split(".")[0]) >= 2026


def repair_unsloth(py, cons, extra, dry):
    """Upgrade Unsloth and its partner packages together under the same constraints. pip prints the exact conflict if impossible."""
    print("\n== Repairing Unsloth ==")
    run([py, "-m", "pip", "install", "--upgrade", "unsloth>=2026.1", "unsloth_zoo>=2026.1", "-c", cons, *extra], dry, cwd=ROOT)


def torch_state(py, dry, need_gpu=True, after_install=False):
    """(works, description) as seen from inside the venv."""
    if dry:
        return (after_install, "dry-run")
    sys.path.insert(0, str(ROOT))
    from ftstudio.doctor import torch_info
    t = torch_info(str(py))
    if t.get("error"):
        return (False, t["error"][:200])
    desc = f"{t['torch']} accelerator={t['accelerator']} {t.get('device') or ''}".strip()
    return (t["accelerator"] != "cpu" if need_gpu else True, desc)


def doctor_json(py) -> dict:
    r = subprocess.run([str(py), "-m", "ftstudio.doctor", "--deep", "--json"], capture_output=True, text=True)
    try:
        return json.loads(r.stdout[r.stdout.index("{"):])
    except ValueError:
        return {"status": "unknown", "message": (r.stdout + r.stderr)[-400:]}


def write_launchers():
    (ROOT / "run.bat").write_text('@echo off\r\ncd /d "%~dp0"\r\nif not exist .venv\\Scripts\\python.exe (\r\n  echo Run install.bat first.\r\n  pause\r\n  exit /b 1\r\n)\r\n'
                                  '.venv\\Scripts\\python.exe -m ftstudio\r\n', encoding="utf-8")
    (ROOT / "run.sh").write_text('#!/bin/sh\ncd "$(dirname "$0")"\n[ -x .venv/bin/python ] || { echo "Run python install.py first."; exit 1; }\nexec .venv/bin/python -m ftstudio\n', encoding="utf-8")
    try:
        (ROOT / "run.sh").chmod(0o755)
    except OSError:
        pass


def install_torch(py, tags, need_gpu, repair, dry):
    ok, info = torch_state(py, dry, need_gpu)
    if ok and not repair:
        print(f"PyTorch already works: {info}")
        return tags[0] if tags else None
    last = info
    for tag in tags:
        print(f"\n== Trying PyTorch build '{tag}' ==")
        try:
            run([py, "-m", "pip", "install", "--upgrade", "--force-reinstall", "torch", "torchvision", "--index-url", INDEX.format(tag)], dry)
        except subprocess.CalledProcessError:
            last = f"pip could not install the '{tag}' wheels"
            continue
        ok, last = torch_state(py, dry, need_gpu, after_install=True)
        if ok:
            print(f"PyTorch works: {last}")
            return tag
        print(f"'{tag}' installed but the GPU is not usable ({last}); trying the next build.")
    raise SystemExit(f"\nCould not get a working PyTorch for this hardware (tried {', '.join(tags) or 'none'}). Last result: {last}\n"
                     "Update the GPU driver and run this again. NVIDIA: check `nvidia-smi`. AMD: check `rocminfo`.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv", default=str(ROOT / ".venv"))
    ap.add_argument("--tag", "--cuda", dest="tag", help="force a PyTorch wheel tag, e.g. cu128, rocm7.2, xpu, cpu")
    ap.add_argument("--repair", action="store_true", help="re-resolve Unsloth and the Hugging Face libraries (keeps a working PyTorch)")
    ap.add_argument("--reinstall-torch", action="store_true", help="reinstall PyTorch even if it already works (large download)")
    ap.add_argument("--no-train", action="store_true", help="core only (prep, synth, eval); no PyTorch")
    ap.add_argument("--cpu", action="store_true", help="install CPU training when no GPU is found (very slow)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    a = ap.parse_args(argv)
    dry, venv = a.dry_run, Path(a.venv)
    py = venv_python(venv)
    if sys.version_info < (3, 10):
        raise SystemExit(f"Python 3.10 or newer is required (this is {sys.version.split()[0]}).")

    hw = detect()
    pl = plan(hw, a.tag, a.cpu, a.no_train)
    print(f"Hardware: {hw.get('name') or 'no supported GPU'} ({hw.get('os')} {hw.get('arch')})\nPlan: {pl['label']}")
    if pl.get("note"):
        print(f"NOTE: {pl['note']}")
    if pl["kind"] == "torch" and not pl["tags"]:
        raise SystemExit(f"Your driver supports only CUDA {hw.get('driver_cuda')}, older than any current PyTorch build. Update the NVIDIA driver.")

    if not py.exists():
        run([sys.executable, "-m", "venv", venv], dry)
    run([py, "-m", "pip", "install", "--upgrade", "pip"], dry)

    extra, cons, tag = [], None, None
    if pl["kind"] == "core":
        run([py, "-m", "pip", "install", "-e", "."], dry, cwd=ROOT)
    else:
        if pl["kind"] == "torch":
            tag = install_torch(py, pl["tags"], pl["need_gpu"], a.reinstall_torch, dry)
            extra = ["--extra-index-url", INDEX.format(tag)]
        else:  # default PyPI wheel (macOS has MPS support built in)
            run([py, "-m", "pip", "install", "torch", "torchvision"], dry)
        cons = venv / "torch-constraints.txt"
        bounds = unsloth_bounds() if pl["extras"] == "train" else []
        if not dry:
            frz = subprocess.run([str(py), "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
            cons.write_text(constraints_text(frz, bounds))
        target = f".[{pl['extras']}]"
        try:  # the pins stop pip from swapping our PyTorch build, and from choosing libraries Unsloth cannot import with
            run([py, "-m", "pip", "install", "-e", target, "-c", cons, *extra], dry, cwd=ROOT)
        except subprocess.CalledProcessError:
            print("\nThe pinned install failed (the conflict is shown above). Retrying with only the Unsloth bounds...")
            cons2 = venv / "bounds-constraints.txt"
            if not dry:
                cons2.write_text(constraints_text([], bounds))
            run([py, "-m", "pip", "install", "-e", target, "-c", cons2, *extra], dry, cwd=ROOT)
            if pl["kind"] == "torch" and not torch_state(py, dry, pl["need_gpu"], after_install=True)[0]:
                run([py, "-m", "pip", "install", "--force-reinstall", "--no-deps", "torch", "torchvision", "--index-url", INDEX.format(tag)], dry)
        if pl.get("extras") == "train" and not dry and (a.repair or not is_modern(unsloth_version(py))):
            repair_unsloth(py, cons, extra, dry)

    if not dry:
        write_launchers()
        print("\n== Final check ==")
        rep = doctor_json(py)
        if pl.get("extras") == "train" and "BROKEN" in str(rep.get("unsloth", "")):
            repair_unsloth(py, cons, extra, dry)
            rep = doctor_json(py)
        print(f"[{rep.get('status', 'unknown')}] {rep.get('message', '')}")
        for k in ("gpu", "accelerator", "torch", "backends", "unsloth", "versions", "note", "fix"):
            if rep.get(k):
                print(f"  {k}: {rep[k]}")
        if rep.get("status") != "ok" and pl["kind"] != "core":
            raise SystemExit(f"\nInstalled, but the hardware check failed (see above). Full log: {LOG}")
    print("\nDone. Start ftstudio with run.bat (Windows) or ./run.sh, then open http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
