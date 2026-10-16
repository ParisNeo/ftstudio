"""Download one GGUF file from Hugging Face into $FTS_HOME/models. params: {"repo": "owner/name", "file": "model-Q4_K_M.gguf"}
Gated repos need `huggingface-cli login` or HF_TOKEN."""
import json
import re
import sys
from pathlib import Path

from ..gguf_catalog import models_dir

REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
FILE = re.compile(r"^[\w.-]+\.gguf$")


def download(repo: str, file: str, dest: Path) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=file, local_dir=str(dest))


def main(run_dir: str):
    run = Path(run_dir)
    p = json.loads((run / "params.json").read_text())
    repo, file = p.get("repo", ""), p.get("file", "")
    if not REPO.match(repo) or ".." in repo or not FILE.match(file):
        raise ValueError(f"need a Hugging Face repo (owner/name) and a plain .gguf file name, got {repo!r} / {file!r}")
    dest = models_dir() / repo.replace("/", "__")
    print(f"downloading {repo}/{file} to {dest}", flush=True)
    try:
        path = download(repo, file, dest)
    except Exception as e:
        raise RuntimeError(f"could not download {repo}/{file}: {e}") from e
    res = {"path": str(Path(path)), "size_mb": round(Path(path).stat().st_size / 1e6), "repo": repo, "file": file}
    (run / "result.json").write_text(json.dumps(res, indent=2))
    print("done:", res["path"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
