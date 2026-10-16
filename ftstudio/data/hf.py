"""Hugging Face datasets as prep sources.
Spec string: owner/name[:config][@split][#limit]   e.g.  HuggingFaceTB/smoltalk:all@train#5000
or a dict {"repo","config","split","limit","revision","format","columns"}.
With a limit the dataset is streamed (only that many rows are downloaded); without one it is downloaded in full.
Gated datasets need `huggingface-cli login` or HF_TOKEN."""
import re

REPO = re.compile(r"^[A-Za-z0-9][\w.-]*(?:/[\w.-]+)?$")
SPEC = re.compile(r"^(?P<repo>[A-Za-z0-9][\w.-]*(?:/[\w.-]+)?)(?::(?P<config>[\w.-]+))?(?:@(?P<split>[\w.\[\]:%-]+))?(?:#(?P<limit>\d+))?$")


def parse_spec(spec) -> dict:
    if isinstance(spec, dict):
        s = dict(spec)
        limit = s.get("limit")
    else:
        m = SPEC.match(str(spec).strip())
        if not m:
            raise ValueError(f"bad dataset spec {spec!r}; use owner/name[:config][@split][#limit]")
        s = {k: v for k, v in m.groupdict().items() if v}
        limit = s.get("limit")
    repo = s.get("repo", "")
    if not REPO.match(repo) or ".." in repo:
        raise ValueError(f"bad Hugging Face dataset name {repo!r} (local paths are not allowed here)")
    s["limit"] = int(limit) if limit else None
    if s["limit"] is not None and s["limit"] < 1:
        raise ValueError("limit must be >= 1")
    s.setdefault("config", None)
    s["split"] = s.get("split") or "train"
    return s


def label(s: dict) -> str:
    return s["repo"] + (f":{s['config']}" if s.get("config") else "") + f"@{s['split']}"


def _default_loader(s: dict):
    from datasets import load_dataset  # lazy: only needed for HF sources

    kw = {"split": s["split"], "streaming": bool(s["limit"])}
    if s.get("revision"):
        kw["revision"] = s["revision"]
    ds = load_dataset(s["repo"], s["config"], **kw) if s["config"] else load_dataset(s["repo"], **kw)
    return ds.take(s["limit"]) if s["limit"] else ds


def iter_records(s: dict, loader=None):
    """Yield (row_no, record|None, error|None) like prep.load_records."""
    print(f"loading {label(s)}" + (f" (first {s['limit']} rows)" if s["limit"] else " (full download)"), flush=True)
    try:
        ds = (loader or _default_loader)(s)
    except Exception as e:
        raise RuntimeError(f"could not load Hugging Face dataset {label(s)}: {e}") from e
    for i, row in enumerate(ds, 1):
        if s["limit"] and i > s["limit"]:
            break
        yield (i, dict(row), None) if hasattr(row, "keys") else (i, None, "row is not a mapping")
