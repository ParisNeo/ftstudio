"""Apple Silicon training with mlx-lm (`pip install "mlx-lm[train]"`): LoRA on the unified-memory GPU via MLX.
Loss is computed on assistant text only through mlx_lm.lora --mask-prompt."""
import json
import math
import re
import subprocess
import sys
import time

from ..train_utils import write_train_result

TRAIN = re.compile(r"Iter (\d+): Train loss ([0-9.]+)")
VAL = re.compile(r"Iter (\d+): Val loss ([0-9.]+)")


def build_command(run, cfg, data_dir, n_train: int) -> list[str]:
    bs = max(1, int(cfg["batch_size"]))
    iters = int(cfg["max_steps"]) if cfg.get("max_steps") else max(1, math.ceil(n_train / bs) * int(math.ceil(cfg["epochs"])))
    return [sys.executable, "-m", "mlx_lm.lora", "--model", cfg["base_model"], "--train", "--data", str(data_dir),
            "--iters", str(iters), "--batch-size", str(bs), "--learning-rate", str(cfg["learning_rate"]),
            "--adapter-path", str(run / "adapter"), "--mask-prompt", "--config", str(run / "mlx_config.yaml")]


def parse_line(line: str):
    m = TRAIN.search(line)
    if m:
        return {"step": int(m[1]), "loss": float(m[2])}
    m = VAL.search(line)
    return {"step": int(m[1]), "eval_loss": float(m[2])} if m else None


def run_mlx(run, cfg, ds, rep):
    data = run / "mlx_data"
    data.mkdir(exist_ok=True)
    rows = ds["train"].read_text(encoding="utf-8").splitlines()
    val = ds["val"].read_text(encoding="utf-8").splitlines() if ds["val"].exists() else []
    (data / "train.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (data / "valid.jsonl").write_text("\n".join(val or rows[-1:]) + "\n", encoding="utf-8")
    (run / "mlx_config.yaml").write_text(json.dumps({"lora_parameters": {"rank": cfg["lora_r"], "dropout": 0.0}}))  # JSON is valid YAML
    cmd = build_command(run, cfg, data, len(rows))
    print("$ " + " ".join(cmd), flush=True)
    metrics, t0 = [], time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
        rec = parse_line(line)
        if rec:
            metrics.append(rec)
            print("METRIC " + json.dumps(rec), flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"mlx_lm.lora exited with code {proc.returncode}; read the log above")
    return write_train_result(run, cfg, ds, metrics, time.time() - t0, None, "mlx", "mlx", run / "adapter")
