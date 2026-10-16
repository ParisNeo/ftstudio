"""What hardware and software stack can train here? Vendors: NVIDIA (CUDA), AMD (ROCm), Intel (XPU), Apple Silicon (MPS/MLX).
CLI: python -m ftstudio.doctor [--deep] [--json]   (--deep also imports Unsloth when it is the chosen engine, which is slow)."""
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys

SNIPPET = """import json
try:
    import torch
    cuda = torch.cuda.is_available()
    hip = getattr(torch.version, "hip", None)
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    acc = ("rocm" if hip else "cuda") if cuda else ("xpu" if xpu else ("mps" if mps else "cpu"))
    dev = torch.cuda.get_device_name(0) if cuda else (torch.xpu.get_device_name(0) if xpu else ("Apple GPU (MPS)" if mps else None))
    print(json.dumps({"torch": torch.__version__, "accelerator": acc, "cuda_available": cuda, "cuda": torch.version.cuda, "hip": hip, "device": dev}))
except Exception as e:
    print(json.dumps({"error": repr(e)}))
"""
FIX_INSTALL = "Close ftstudio, run install.bat (or: python install.py) to build a .venv with the right PyTorch for your hardware, then start it with run.bat (or ./run.sh)."
FIX_BACKEND = "Run install.bat (or: python install.py) again so the training packages for your hardware are installed."


def parse_cuda(text: str):
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", text or "")
    return m.group(1) if m else None


def _run(cmd, timeout=20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    line = (_run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"]).strip().splitlines() or [""])[0]
    name, _, drv = (x.strip() for x in line.partition(","))
    return {"gpu": name or "NVIDIA GPU", "driver": drv, "cuda": parse_cuda(_run([exe]))}


def vendor_of(name: str):
    n = name.lower()
    if re.search(r"nvidia|geforce|quadro|tesla|rtx", n):
        return "nvidia"
    if re.search(r"\bamd\b|radeon|advanced micro devices|instinct|\bati\b", n):
        return "amd"
    if "intel" in n and re.search(r"\barc\b|iris xe|data center gpu|\bmax\b", n):
        return "intel"             # integrated UHD graphics is not a training device
    return None


def _gpu_names(system: str) -> list[str]:
    if system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"], 30)
    elif system == "Linux":
        out = "\n".join(l for l in _run(["lspci"]).splitlines() if re.search(r"VGA|3D|Display", l))
    else:
        out = ""
    return [l.strip() for l in out.splitlines() if l.strip()]


def detect_hardware() -> dict:
    system, machine = platform.system(), platform.machine().lower()
    base = {"os": system, "arch": machine, "vendor": None, "name": None, "driver": None, "driver_cuda": None}
    smi = nvidia_smi()
    if smi:
        return {**base, "vendor": "nvidia", "name": smi["gpu"], "driver": smi["driver"], "driver_cuda": smi["cuda"]}
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return {**base, "vendor": "apple", "name": _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or "Apple Silicon"}
    for n in _gpu_names(system):
        v = vendor_of(n)
        if v:
            return {**base, "vendor": v, "name": n}
    return base


def torch_info(python=None, timeout=120) -> dict:
    try:
        r = subprocess.run([python or sys.executable, "-c", SNIPPET], capture_output=True, text=True, timeout=timeout)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as e:
        return {"error": f"could not run python to check torch: {e!r}"}


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def diagnose(hw: dict, t: dict, mlx_ok: bool = False) -> dict:
    vendor, name = hw.get("vendor"), hw.get("name")
    acc = None if t.get("error") else t.get("accelerator") or ("cuda" if t.get("cuda_available") else "cpu")
    if acc and acc != "cpu":
        return {"status": "ok", "message": f"PyTorch {t['torch']} can use {t.get('device')} ({acc}).", "fix": ""}
    if vendor == "apple" and mlx_ok:
        return {"status": "ok", "message": f"{name}: MLX is installed (native Apple Silicon training).", "fix": ""}
    if vendor is None:
        return {"status": "no_gpu", "fix": "", "message": "No supported GPU was found (NVIDIA, AMD, Intel Arc or Apple Silicon). Prepare data, synth and evaluation still run; training needs a GPU, or set allow_cpu: true for (very slow) CPU training."}
    if t.get("error"):
        if "No module named" in t["error"]:
            return {"status": "no_torch", "message": "PyTorch is not installed in this Python environment.", "fix": FIX_INSTALL}
        return {"status": "torch_broken", "message": f"PyTorch fails to import: {t['error'][:300]}", "fix": FIX_INSTALL}
    v = t.get("torch", "?")
    if vendor == "apple":
        return {"status": "cpu_torch", "message": f"{name}: neither MLX nor PyTorch MPS is available in this environment.", "fix": FIX_INSTALL}
    if "+cpu" in v or not (t.get("cuda") or t.get("hip") or "xpu" in v):
        return {"status": "cpu_torch", "message": f"PyTorch {v} is the CPU-only build, but this machine has {name}. Training cannot use the GPU.", "fix": FIX_INSTALL}
    return {"status": "cuda_unusable", "message": f"PyTorch {v} has {vendor.upper()} support but cannot use the GPU ({name}).",
            "fix": "Update the GPU driver (NVIDIA/AMD/Intel), or run python install.py --reinstall-torch to pick a PyTorch build that matches it."}


def package_versions() -> dict:
    from importlib import metadata
    out = {}
    for n in ("unsloth", "unsloth_zoo", "transformers", "datasets", "trl", "peft", "accelerate", "bitsandbytes", "torch"):
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return out


def available_backends(rep: dict, has=has_module, allow_cpu=False) -> list[str]:
    acc, out = rep.get("accelerator"), []
    gpu = acc in ("cuda", "rocm", "xpu", "mps")
    if acc in ("cuda", "rocm", "xpu") and has("unsloth"):
        out.append("unsloth")
    if (gpu or allow_cpu) and has("peft") and has("transformers"):
        out.append("peft")
    if rep.get("vendor") == "apple" and has("mlx_lm"):
        out.append("mlx")
    return out


def report(python=None, hw_fn=detect_hardware, torch_fn=torch_info, mlx_fn=lambda: has_module("mlx") and has_module("mlx_lm"), has=has_module) -> dict:
    hw, t = hw_fn(), (torch_fn(python) if python else torch_fn())
    r = {**diagnose(hw, t, bool(mlx_fn())), "python": python or sys.executable, "in_venv": sys.prefix != sys.base_prefix,
         "torch": t.get("torch"), "accelerator": None if t.get("error") else t.get("accelerator"), "device": t.get("device"),
         "vendor": hw.get("vendor"), "gpu": hw.get("name"), "driver": hw.get("driver"), "driver_cuda": hw.get("driver_cuda"),
         "os": hw.get("os"), "arch": hw.get("arch")}
    r["backends"] = available_backends(r, has) if python is None else []
    return r


def choose_backend(requested: str, rep: dict, has=has_module, allow_cpu=False) -> str:
    avail = available_backends(rep, has, allow_cpu)
    if requested and requested != "auto":
        if requested not in avail:
            raise RuntimeError(f"training engine {requested!r} is not usable here (usable: {avail or 'none'}). {FIX_BACKEND}")
        return requested
    acc = rep.get("accelerator")
    order = {"cuda": ["unsloth", "peft"], "rocm": ["peft", "unsloth"], "xpu": ["peft", "unsloth"]}.get(acc) or ["mlx", "peft"]
    for b in order:
        if b in avail:
            return b
    raise RuntimeError(f"No training engine is installed for this hardware (installed and usable: {avail or 'none'}). {FIX_BACKEND}")


def require_accelerator(allow_cpu: bool = False, **kw) -> dict:
    """Called by the train worker BEFORE importing any training library: fails with the fix, not a traceback."""
    r = report(**kw)
    if r["status"] != "ok":
        if allow_cpu and r["status"] in ("no_gpu", "cpu_torch"):
            print("WARNING: no usable GPU; allow_cpu is set, so training runs on the CPU. This is very slow.", flush=True)
            return r
        raise RuntimeError(f"{r['message']} {r['fix']}".strip())
    print(f"Hardware check: {r['message']}", flush=True)
    return r


require_cuda = require_accelerator  # old name


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    r = report()
    if "--deep" in argv and r["status"] == "ok" and "unsloth" in r["backends"] and r.get("accelerator") == "cuda":
        try:
            import unsloth  # noqa: F401
            import transformers
            r["unsloth"], r["transformers"] = "imports", transformers.__version__
            if int(transformers.__version__.split(".")[0]) < 5:
                r["note"] = "transformers < 5: Qwen3.5 models need v5."
        except Exception as e:
            r["unsloth"] = f"BROKEN: {e!r}"[:300]
            r["backends"] = [b for b in r["backends"] if b != "unsloth"]
            if r["backends"]:
                r["note"] = "Unsloth cannot be imported, so training falls back to the generic peft engine (slower, more memory). Run python install.py --repair to fix Unsloth."
            else:
                r.update(status="unsloth_broken", message=f"Unsloth failed to import: {e!r}"[:400], fix=FIX_INSTALL)
    if "--deep" in argv:
        r["versions"] = package_versions()
    if "--json" in argv:
        print(json.dumps(r, indent=2))
    else:
        print(f"[{r['status']}] {r['message']}")
        for k in ("python", "os", "arch", "gpu", "vendor", "accelerator", "torch", "driver", "driver_cuda", "backends", "unsloth", "transformers", "versions", "note"):
            if r.get(k):
                print(f"  {k}: {r[k]}")
        if r["fix"]:
            print("  fix:", r["fix"])
    return 0 if r["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
