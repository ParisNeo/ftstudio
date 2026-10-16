# ftstudio

Own-built fine-tuning studio. **Built so far: job runner, eval rig, data prep.** Pipeline complete end to end: prep -> train -> export -> eval, all queued jobs, UI for each. **Train/export are untested on a real GPU** (see below).

```
pip install -e ".[dev,eval]"
python -m ftstudio                      # API on http://127.0.0.1:8000  (docs at /docs)
pytest                                  # 35 tests, no GPU needed (fake llama binaries)
```

## Design
- `db.py`: SQLite job table. `runner.py`: **one loop = one GPU lock**; jobs run serially, each as a subprocess (`python -m ftstudio.workers.<kind> runs/<id>`), so OOMs/leaks die with the subprocess.
- Worker contract: read `params.json`, print to stdout (-> `log.txt`, streamed over WebSocket), write `result.json`.
- API: `POST /jobs`, `GET /jobs[/id]`, `POST /jobs/id/cancel`, `GET /jobs/id/log?offset=`, `GET /jobs/id/artifacts/<file>`, `WS /jobs/id/stream`. `/` serves the UI (`ftstudio/static/index.html`, single file, no build step); API docs at `/docs`.
- Cancel kills the whole process tree (killpg on POSIX, `taskkill /T` on Windows), including llama-server.

## Presets and the start page
The UI opens on a four-step "Start here" guide and every form has plain-language help plus a **Preset** menu (smoke test, balanced for 12 GB, thorough, low memory, export sizes...). Failed jobs show a "What to try" box for common errors (out of memory, gated model, missing package, wrong path). Add your own presets in `$FTS_HOME/presets.json`: a list of `{"id","kind","name","desc","params"}`; the same id overrides a built-in. `GET /presets` lists them.

## Prepare a dataset (`kind: "prep"`)
Inputs: `.jsonl .json .csv .tsv`. Per-record format detection: **messages**, **sharegpt**, **alpaca**, **prompt/completion** (also question/answer, input/output, query/response), or an explicit `columns` map for CSV/odd schemas.

```
{"kind":"prep","params":{
  "inputs":["/data/a.jsonl","/data/b.csv"], "columns":{"user":"q","assistant":"a"},
  "tokenizer":"meta-llama/Llama-3.1-8B-Instruct", "max_seq_len":2048,
  "val_fraction":0.05, "decontaminate_against":["/data/my_tasks.jsonl"]}}
```
Pipeline: normalize roles -> validate turn order (must end on assistant) -> **decontaminate against your eval files** -> dedupe (whitespace/case-normalized) -> render with the model's chat template + count tokens -> drop (**never truncate**) over-long -> deterministic split (hash of the first user prompt: same prompt always lands in the same split, stable as data grows).
Outputs: `train.jsonl`, `val.jsonl` (`{"messages":[...]}`), `report.json` (counts, drop reasons with examples, length stats, dataset sha256), `preview.txt` (rendered samples), and a **template probe** (prefix / between / suffix) that the train worker will use as response-only masking anchors.
Not handled: raw documents (PDF/TXT -> Q&A needs a synthetic-generation step, separate worker).

## Train (`kind: "train"`) and export (`kind: "export"`)
```
{"kind":"train","params":{"dataset":{"job":"<prep job id>"},"base_model":"unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit","max_seq_len":2048,"epochs":1,"lora_r":16}}
{"kind":"export","params":{"train_job":"<train job id>","quantizations":["q4_k_m"],"auto_eval":{"custom":{"file":"/data/my_tasks.jsonl"}}}}
```
Train: QLoRA via Unsloth + TRL, loss on assistant tokens only (anchors come from the prep template probe), adapter saved to `adapter/`, loss curve in the UI. Export: merge + GGUF + quantize via Unsloth/llama.cpp, then **one eval job is queued per GGUF** (serial runner starts it after export exits). Install GPU deps with `pip install -e ".[train]"`.

## Install (Windows, Linux, macOS) and hardware support
```
python install.py        # Windows: install.bat    Linux/macOS: ./install.sh
run.bat                  # Linux/macOS: ./run.sh   -> http://127.0.0.1:8000
```
`install.py` creates `.venv`, detects your hardware, installs the PyTorch build that matches it, installs the training engine for it, runs a check, and writes the launchers. Always start ftstudio from the venv (run.bat / run.sh): jobs use the same Python as the server.

| Hardware | PyTorch build (tags tried newest first) | Training engine (`backend: auto`) | Export to GGUF |
|---|---|---|---|
| NVIDIA | CUDA: cu132, cu130, cu128, cu126, cu124, never newer than the driver supports, each verified with `torch.cuda.is_available()` | `unsloth` (falls back to `peft`) | Unsloth |
| AMD, Linux | ROCm: rocm7.2, rocm7.1, rocm6.4 (PyTorch ROCm wheels are Linux-only; on Windows use WSL2) | `peft` (`unsloth` selectable) | llama.cpp tools |
| Intel Arc | XPU (`xpu` index) | `peft` | llama.cpp tools |
| Apple Silicon | default PyPI wheel (MPS) + `mlx-lm` | `mlx` (falls back to `peft` on MPS) | llama.cpp tools |
| no GPU | core only (prep, synth, eval); `--cpu` adds CPU training | `peft` with `allow_cpu: true` (very slow) | llama.cpp tools |
| Intel/AMD iGPU, Vulkan | n/a | none: Vulkan only accelerates GGUF *inference* | n/a |

Engines: `unsloth` (fastest, NVIDIA-first), `peft` (transformers + PEFT LoRA on CUDA/ROCm/XPU/MPS/CPU; 4-bit only with bitsandbytes on NVIDIA/AMD, otherwise 16-bit LoRA, fp32 on MPS/CPU), `mlx` (`mlx_lm.lora --mask-prompt` on Apple Silicon). Pick one with `backend` in the train params, or leave it on auto.
**Export for peft/mlx** needs llama.cpp: clone it, set `FTS_LLAMA_CPP` to the folder (for `convert_hf_to_gguf.py`; `pip install -r requirements/requirements-convert_hf_to_gguf.txt`), and have `llama-quantize` on PATH or in `FTS_LLAMA_QUANTIZE`. **Inference** (eval, synth) uses your llama.cpp `llama-server`: use the build for your hardware (Metal on Mac; CUDA, HIP, Vulkan or SYCL elsewhere) via `FTS_LLAMA_SERVER`.
Options: `--tag cu128|rocm7.2|xpu|cpu` force a build, `--repair` re-resolve Unsloth and the Hugging Face libraries (keeps a working PyTorch), `--reinstall-torch` reinstall PyTorch (large download), `--no-train` core only, `--cpu` CPU training, `--dry-run` print the plan. `torchaudio` is not installed (not needed). Why the installer exists: on Windows a plain `pip install torch` is the **CPU-only** build and training fails with "torch.cuda.is_available() is False". Train now checks the hardware first and stops with the fix; the UI shows a banner; `GET /doctor` or `python -m ftstudio.doctor` gives the diagnosis.
**Troubleshooting.** Everything the installer runs is logged to `install.log` (paste it when something fails). `python -m ftstudio.doctor --deep` lists the installed versions. *"cannot import name 'ConstantLengthDataset' from trl"*: Unsloth and TRL/transformers versions are out of step (pip fell back to a 2025 Unsloth because the Hugging Face libraries it found were newer than the current Unsloth allows). The installer now constrains transformers, datasets, trl, peft, accelerate and bitsandbytes to the ranges Unsloth itself declares on PyPI, refuses an Unsloth older than 2026, and repairs once automatically; run `python install.py --repair` for an existing venv. Meanwhile training with `backend: auto` falls back to the peft engine if Unsloth cannot be imported.
**Tested here (Linux, no GPU):** plans and dry runs per vendor, hardware diagnosis, engine selection, response masking with a real tokenizer, the MLX command and log parsing, the llama.cpp export pipeline (fakes), and a real venv install of the core. **Never run on real NVIDIA, AMD, Intel or Apple hardware.** Unverified details: MLX `lora_parameters.rank` config key and log-line format; `--mask-prompt` on multi-turn chats probably trains only the last assistant turn (check the log); `AutoModelForCausalLM` on multimodal checkpoints (Qwen3.5, Gemma 4); PyTorch XPU wheels on Windows.

## Fine-tune a model on your own library (example: ascii_colors)
1. **Prep:** preset "Your library: ascii_colors discussions". It reads `ParisNeo/ascii_colors_discussions` (columns `system_prompt` + `discussion`, auto-detected) and replaces the per-row system prompts, which say "based only on this conversation", with one neutral expert prompt.
2. **Baseline:** run `examples/ascii_colors_eval.jsonl` on the untuned model. Its `match: "python"` mode **runs** the generated code and checks the output, so invented APIs fail. Every reference solution in it is verified against the real package by `tests/test_library_workflow.py`. It executes model-written code unsandboxed: your machine, your prompts.
3. **Train:** preset "Your library: ascii_colors on Qwen3.5 2B". Watch validation loss; with 36 conversations, overfitting is the main risk.
4. **Export** with `auto_eval` pointing at the same file and compare to the baseline.
Tiny data teaches habits, not exhaustive recall. For "perfect", grow the data (more Q&A and verified code examples generated from the library's source and docs) and keep the README retrievable at inference time.

## Generate training entries from raw docs (`kind: "synth"`)
Uses a GGUF (started with llama.cpp for the task) or any running OpenAI-compatible server to turn documents (.md .rst .txt .py, folders searched recursively) into entries saved in a **named library** (`$FTS_HOME/library/<name>.jsonl`). `mode: "append"` adds to a library, `"create"` starts a new one (refuses if it has entries unless `overwrite: true`). Prepare-data reads libraries via `libraries: ["name"]`.
```
{"kind":"synth","params":{"gguf":"/models/Qwen3.5-4B-Q4_K_M.gguf","sources":["/projects/ascii_colors/docs"],"library":"ascii_colors_docs",
  "must_import":"ascii_colors","verify_code":true,"judge":true,"dry_run":true}}
```
**Your prompt:** `prompt` replaces the built-in generation prompt (string or list of lines; must contain `{chunk}`, may use `{n}` `{source}`); `generator_system` replaces the system message; `system_prompt` is the system turn stored in each entry.
**What makes entries trustworthy (filters, not guarantees):** too-short / "the excerpt" questions dropped; duplicates (exact or 85% word overlap, also against the existing library); optional decontamination against your eval files; **every ```python block is executed** (must exit 0 and, with `must_import`, import your module); failures are sent back to the model with the error for `repair_rounds` fixes, then dropped; a judge pass checks every claim against the source passage (self-judging: the same model that wrote it, so weaker than a human or a bigger model; use a stronger `server_url` for the judge-quality you need).
**Workflow:** run the "Dry run" preset (3 passages, nothing saved), read `preview.jsonl` **and** `rejects.jsonl`, tune the prompt, then run for real. Browse a library at `/library/<name>`; delete a bad entry with `DELETE /library/<name>/<index>`.
Executing generated code is unsandboxed: your machine, your documents.
**Untested:** the real llama-server path (tests use a scripted model); thinking-model `<think>` output is stripped.

## GGUF models for testing
The Test and Generate forms list the `.gguf` files on this computer (`$FTS_HOME/models`, finished exports, and any folders in `FTS_MODELS_DIR`) and offer example models to **download with one click** (a `fetch` job: `{"repo": "unsloth/Qwen3.5-2B-GGUF", "file": "Qwen3.5-2B-Q4_K_M.gguf"}`). Add your own examples in `$FTS_HOME/gguf_examples.json`. The first examples are Qwen3.5 2B and Gemma 4 E2B at Q4_K_M and Q8_0 (file names verified on their Hugging Face pages); the 4B and 0.8B names follow the same pattern but were not individually checked. Qwen3.5 and Gemma 4 need a **recent llama.cpp**. To compare fairly, test the original at the same quantization you export (q4_k_m).

## Hugging Face datasets in prep
`hf_datasets` takes specs `owner/name[:config][@split][#limit]` (or dicts with `repo, config, split, limit, revision, format, columns`), alone or mixed with local `inputs`:
```
{"kind":"prep","params":{"hf_datasets":["yahma/alpaca-cleaned#2000","HuggingFaceTB/smoltalk:all@train#5000"],"tokenizer":"unsloth/Qwen3.5-2B","max_seq_len":2048}}
```
With `#limit` the dataset is **streamed** (only that many rows are downloaded); without it, it is downloaded whole. Formats are auto-detected per row (Alpaca incl. Dolly's `context`, ShareGPT `conversations`, `messages`, question/answer...). Known datasets are suggested as buttons in the form (`/datasets`; extend with `$FTS_HOME/datasets.json`). Gated datasets need `huggingface-cli login` or `HF_TOKEN`. Local paths are rejected in specs. Needs `pip install -e ".[hf]"`. The report lists rows read per source. **Read each dataset card for its license**; revisions are not pinned unless you pass `revision`.

## Eval a GGUF
Env: `FTS_HOME` (data dir, default `./data`), `FTS_LLAMA_SERVER`, `FTS_LLAMA_PERPLEXITY` (binary paths).

```
curl -X POST localhost:8000/jobs -H 'content-type: application/json' -d '{
  "kind": "eval",
  "params": {
    "gguf": "/models/base-Q4_K_M.gguf",
    "suites": {
      "perplexity": {"file": "/data/wiki.test.raw", "ctx": 512},
      "custom":     {"file": "/data/my_tasks.jsonl"},
      "lm_eval":    {"tasks": ["gsm8k"], "limit": 100, "num_fewshot": 5, "tokenizer": "meta-llama/Llama-3.1-8B-Instruct"}
    }}}'
```
Custom JSONL: `{"id","prompt"|"messages","expected","match": exact|contains|regex|numeric,"tag"}`.
**Always run the base model first.** That is the control.

## UNVERIFIED against real binaries (tests use fakes) — check on first run
-2. HF loading ran only against a fake loader and local files (no network here). The `datasets` calls (`streaming=True`, `.take(n)`) were checked on a local JSONL. Check one real download (the Alpaca preset) first.
-1. **Train/export never ran on a GPU.** API used: `FastLanguageModel.from_pretrained/get_peft_model`, `unsloth.chat_templates.train_on_responses_only` (name from memory), TRL `SFTTrainer/SFTConfig` (arg names vary by version; `sft_config` retries known renames), `save_pretrained_gguf(dir, tok, quantization_method=...)` (documented in TRL's Unsloth page). Pin versions once it works.
-1b. On the first run read the log lines `FIRST INPUT` (BOS doubled?) and `TRAINED ON` (must be assistant text only). Both are silent-failure traps.
-1c. Export loads the adapter in 4-bit and merges; if Unsloth rejects that for your setup, merge in 16-bit and convert with llama.cpp's `convert_hf_to_gguf.py` instead.
0. Prep is tested with a stub and a hand-built ChatML-style HF tokenizer, **not** with real Llama/Qwen/Mistral templates. Templates that reject a system role show up as `template_error` drops; templates that inject a default system prompt show up in the probe `prefix`. Read `preview.txt` before training.
1. `llama-perplexity` prints `Final estimate: PPL = X +/- Y` (regex in `workers/eval.py`; fails loudly if not).
2. lm-eval `local-completions` args: `base_url` as full `/v1/completions` endpoint, `tokenizer`/`tokenizer_backend` needed. Verify against your installed lm-eval version.
3. Loglikelihood tasks (MMLU, HellaSwag, ARC) need `echo=true` + logprobs from llama-server's completions endpoint. Unconfirmed for your build. **Start with generative tasks (gsm8k)** and only then try loglikelihood ones.
4. `llama-server` flags `-m --host --port -c -ngl` and `/health` endpoint.
