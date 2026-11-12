#!/usr/bin/env python3
"""ftstudio installer for Windows, Linux and macOS. Creates .venv, detects your hardware, installs the PyTorch build that matches it
(plain `pip install torch` gives the CPU-only build on Windows), installs the training engine for that hardware, verifies it, and
writes run.bat / run.sh.

  NVIDIA  -> CUDA PyTorch (newest tag your driver supports) + Unsloth        AMD (Linux) -> ROCm PyTorch + PEFT
  Intel Arc -> XPU PyTorch + PEFT                                           Apple Silicon -> MLX (mlx-lm) + PyTorch MPS
  no GPU  -> core only (prep, synth, eval), or CPU training with --cpu
  python install.py [--venv .venv] [--tag cu128|rocm7.2|xpu|cpu] [--special-python <path>] [--repair] [--reinstall-torch] [--no-train] [--cpu] [--dry-run]"""
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
# Each entry is (tag, torch_version). Recent PyTorch releases dropped old CUDA tags from the index, and the newest
# releases have no wheel for Python 3.14 yet, so both a tag and a torch version must be picked.
CUDA_BUILDS = [
    ("cu132", "2.9.0"), ("cu130", "2.9.0"), ("cu128", "2.7.1"),
    ("cu126", "2.6.0"), ("cu124", "2.5.1"), ("cu121", "2.5.1"),
]
# PyTorch dropped cu121 entirely (no cp311/cp312/cp313 wheels on that index any more) and cu124 stops at 2.4/2.5; the
# pip forward-compat runtime raises the *user-mode* CUDA to 12.6+, so the survivor building block is the cu126 wheel set.
# We also need the torchvision pin to move in lockstep with torch (torchvision 0.20.1 pairs with torch 2.5.1, 0.21.0 with torch 2.6.0).
TORCHVISION_PIN = {"2.9.0": "0.24.0", "2.7.1": "0.22.1", "2.6.0": "0.21.0", "2.5.1": "0.20.1"}
CUDA_TAGS = [t for t, _ in CUDA_BUILDS]   # newest first; each is tried until PyTorch really sees the GPU
ROCM_TAGS = ["rocm7.2", "rocm7.1", "rocm6.4"]              # Linux only
XPU_TAGS = ["xpu"]
# CUDA 12.x user-mode runtime wheels usable as a pip-only "forward compatibility" shim on an older 12.2 driver.
FORWARD_COMPAT_PKGS = [
    "nvidia-cuda-runtime-cu12==12.6.77",
    "nvidia-cuda-nvrtc-cu12==12.6.77",
    "nvidia-cuda-cupti-cu12==12.6.80",
    "nvidia-cublas-cu12==12.6.4.1",
    "nvidia-cudnn-cu12==9.5.1.17",
    "nvidia-cufft-cu12==11.3.0.4",
    "nvidia-curand-cu12==10.3.7.77",
    "nvidia-cusolver-cu12==11.7.1.2",
    "nvidia-cusparse-cu12==12.5.4.2",
]
# Python versions for which PyTorch publishes cpXXX wheels on our CUDA/CPU indexes. Anything newer (e.g. 3.14) has none,
# and we redirect the user to a supported interpreter before touching the venv.
TORCH_PY = {(3, 10), (3, 11), (3, 12), (3, 13)}


def torch_versions(tag) -> list[str]:
    """torch version pins to try for a given index tag, newest first. Passes through anything not in our table (rocm/xpu/cpu)."""
    return [v for t, v in CUDA_BUILDS if t == tag] or [None]


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


def forward_compat(driver_cuda) -> bool:
    """True when the driver is CUDA 12.x but older than every wheel we know (e.g. 12.2). Those drivers still run the
    cu12x runtime wheels installed by pip, so we can shim forward instead of asking for an admin-only driver update."""
    if not driver_cuda:
        return False
    major, minor = map(int, driver_cuda.split("."))
    return major == 12 and any(tag_version(t) > (major, minor) for t in CUDA_TAGS)


def torch_supported_python() -> bool:
    return sys.version_info[:2] in TORCH_PY


def find_torch_python() -> Path | None:
    """A Python 3.10-3.13 interpreter to build the venv with, when the launcher used an unsupported one (e.g. 3.14).
    Tries py -3.X, python3.X on PATH, and the default per-user install locations on Windows/macOS."""
    for minor in (13, 12, 11, 10):
        for cand in ([["py", f"-3.{minor}"]] if os.name == "nt" else []) + [[f"python3.{minor}"]]:
            try:
                r = subprocess.run(cand + ["-c", "import sys;print(sys.executable)"], capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if r.returncode == 0 and r.stdout.strip():
                return Path(r.stdout.strip())
    if os.name == "nt":
        for minor in (13, 12, 11, 10):
            p = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / f"Python3{minor}" / "python.exe"
            if p.exists():
                return p
    return None


def python_tag(py) -> str:
    """cpXXX wheel tag for a given interpreter (e.g. cp311)."""
    r = subprocess.run([str(py), "-c", "import sys;print(f'cp{sys.version_info[0]}{sys.version_info[1]}')"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "cp311"


def wheel_available(tag: str, version: str, pytag: str) -> bool:
    """True iff PyTorch publishes a wheel for (tag, version, this interpreter's cpXXX tag). Probes the index page once."""
    if version is None:
        return True
    url = f"https://download.pytorch.org/whl/{tag}/torch/"
    try:
        html = urllib.request.urlopen(url, timeout=20).read().decode("utf-8", "replace")
    except Exception:
        return True  # can't probe; let pip decide
    pat = re.compile(rf"torch-{re.escape(version)}[^\"']*?-{pytag}-{pytag}-win_amd64\.whl")
    return pat.search(html) is not None


def forward_compat_tag(driver_cuda) -> str:
    """The wheel tag to install when using the pip forward-compat runtime on an older 12.x driver.
    PyTorch has since removed cu121/cu124 from its index and keeps only cu126+ as long-term builds; the pip shim
    raises the user-mode CUDA to 12.6+, so cu126 is the safe target even on a 12.2 driver."""
    major, minor = map(int, driver_cuda.split("."))
    if major == 12 and minor < 6:
        return "cu126"
    return next((t for t in reversed(CUDA_TAGS) if tag_version(t) >= (major, minor)), CUDA_TAGS[-1])


def plan(hw: dict, forced=None, cpu_ok=False, no_train=False) -> dict:
    """What to install for this hardware. kind: core | torch (index tags to try) | default (plain PyPI torch, Apple)."""
    v, name = hw.get("vendor"), hw.get("name")
    if no_train:
        return {"kind": "core", "label": "core only (prep, synth, eval)", "note": ""}
    if v == "nvidia":
        tags = candidates(hw.get("driver_cuda"), forced)
        if not tags and forward_compat(hw.get("driver_cuda")):
            tags = [forward_compat_tag(hw.get("driver_cuda"))]  # latest wheel the pip forward-compat runtime can extend to
        extras = "train" if tags else "train-generic"
        note = ""
        if forward_compat(hw.get("driver_cuda")) and not forced:
            note = (f"Your NVIDIA driver reports CUDA {hw.get('driver_cuda')}, older than any current PyTorch wheel. "
                    f"Installing the pip forward-compatibility runtime (no admin, no reboot) and the {tags[0]} PyTorch build; "
                    "Unsloth is skipped because its Triton kernels expect a newer driver.")
        return {"kind": "torch", "tags": tags, "extras": extras, "need_gpu": True, "note": note,
                "label": f"NVIDIA {name}, driver supports CUDA {hw.get('driver_cuda')}: CUDA PyTorch{'+ Unsloth' if extras == 'train' else ' (forward-compat, no Unsloth)'}"}
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


def install_forward_compat(py, dry):
    """pip-only NVIDIA runtime shim so an older 12.x driver can load a newer CUDA user-mode path. No admin needed."""
    print("\n== Installing pip forward-compatibility CUDA runtime (User Space Shim) ==")
    run([py, "-m", "pip", "install", *FORWARD_COMPAT_PKGS], dry)


def install_torch(py, tags, need_gpu, repair, dry, driver_cuda=None):
    ok, info = torch_state(py, dry, need_gpu)
    if ok and not repair:
        print(f"PyTorch already works: {info}")
        return tags[0] if tags else None
    last = info
    shimmed = False
    pytag = "cp0" if dry else python_tag(py)
    for tag in tags:
        if forward_compat(driver_cuda) and tag != "cpu" and not shimmed:
            install_forward_compat(py, dry)
            shimmed = True
        for ver in torch_versions(tag):
            if not dry and not wheel_available(tag, ver, pytag):
                print(f"\n== Skipping PyTorch build '{tag}' {ver}: no {pytag} wheel on the {tag} index ==")
                last = f"no {pytag} wheel for torch {ver} on the '{tag}' index"
                continue
            print(f"\n== Trying PyTorch build '{tag}'{' ' + ver if ver else ''} ==")
            if ver:
                tv = TORCHVISION_PIN.get(ver, f"{ver.rsplit('.', 1)[0]}.0")
                pin = [f"torch=={ver}+{tag}", f"torchvision=={tv}+{tag}"]
            else:
                pin = ["torch", "torchvision"]
            try:
                run([py, "-m", "pip", "install", "--upgrade", "--force-reinstall", *pin, "--index-url", INDEX.format(tag)], dry)
            except subprocess.CalledProcessError:
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
                     "Your drivers may be too old. Consider using a user-mode fallback tag or running with --cpu.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv", default=str(ROOT / ".venv"))
    ap.add_argument("--tag", "--cuda", dest="tag", help="force a PyTorch wheel tag, e.g. cu128, rocm7.2, xpu, cpu")
    ap.add_argument("--special-python", help="path to a specific Python interpreter (e.g., user-installed Python 3.12) to bypass system limitations")
    ap.add_argument("--repair", action="store_true", help="re-resolve Unsloth and the Hugging Face libraries (keeps a working PyTorch)")
    ap.add_argument("--reinstall-torch", action="store_true", help="reinstall PyTorch even if it already works (large download)")
    ap.add_argument("--no-train", action="store_true", help="core only (prep, synth, eval); no PyTorch")
    ap.add_argument("--cpu", action="store_true", help="install CPU training when no GPU is found (very slow)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    a = ap.parse_args(argv)
    dry, venv = a.dry_run, Path(a.venv)

    # Resolve interpreter target
    if a.special_python:
        sys.executable = a.special_python
        print(f"Using explicitly requested special Python interpreter: {sys.executable}")

    py = venv_python(venv)
    if sys.version_info < (3, 10):
        raise SystemExit(f"Python 3.10 or newer is required (this is {sys.version.split()[0]}).")

    hw = detect()
    pl = plan(hw, a.tag, a.cpu, a.no_train)
    print(f"Hardware: {hw.get('name') or 'no supported GPU'} ({hw.get('os')} {hw.get('arch')})\nPlan: {pl['label']}")
    if pl.get("note"):
        print(f"NOTE: {pl['note']}")
    
    if pl["kind"] == "torch" and not torch_supported_python():
        alt = Path(a.special_python) if a.special_python else find_torch_python()
        if not alt or not alt.exists():
            raise SystemExit(f"Python {sys.version.split()[0]} has no PyTorch wheels yet (PyTorch supports 3.10-3.13). "
                             "Because your system driver cannot be updated, please install Python 3.12 (per-user, no admin needed) "
                             "and rerun with: python install.py --special-python <path-to-python3.12>")
        print(f"NOTE: Python {sys.version.split()[0]} has no PyTorch wheels. Using special interpreter {alt} for the .venv.")
        sys.executable = str(alt)

    if pl["kind"] == "torch" and not pl["tags"]:
        if forward_compat(hw.get("driver_cuda")):
            raise SystemExit(f"Your driver supports only CUDA {hw.get('driver_cuda')}. The pip forward-compatibility path is unavailable "
                             f"({py} -m pip install {' '.join(FORWARD_COMPAT_PKGS)} failed). Try providing a special Python environment or run with --cpu.")
        raise SystemExit(f"Your driver supports only CUDA {hw.get('driver_cuda')}, older than any current PyTorch build. "
                         "Since you cannot update system drivers, use --special-python or pass --cpu to run on CPU.")

    if not py.exists():
        run([sys.executable, "-m", "venv", venv], dry)
    run([py, "-m", "pip", "install", "--upgrade", "pip"], dry)

    extra, cons, tag = [], None, None
    if pl["kind"] == "core":
        run([py, "-m", "pip", "install", "-e", "."], dry, cwd=ROOT)
    else:
        if pl["kind"] == "torch":
            tag = install_torch(py, pl["tags"], pl["need_gpu"], a.reinstall_torch, dry, hw.get("driver_cuda"))
            extra = ["--extra-index-url", INDEX.format(tag)]
        else:
            run([py, "-m", "pip", "install", "torch", "torchvision"], dry)
        cons = venv / "torch-constraints.txt"
        bounds = unsloth_bounds() if pl["extras"] == "train" else []
        if not dry:
            frz = subprocess.run([str(py), "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
            cons.write_text(constraints_text(frz, bounds))
        target = f".[{pl['extras']}]"
        try:
            run([py, "-m", "pip", "install", "-e", target, "-c", cons, *extra], dry, cwd=ROOT)
        except subprocess.CalledProcessError:
            print("\nThe pinned install failed. Retrying with only the Unsloth bounds...")
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