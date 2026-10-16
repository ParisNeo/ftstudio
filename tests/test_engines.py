import json
import sys
from pathlib import Path

import pytest

from ftstudio import db, library  # noqa: F401
from ftstudio.data import masking
from ftstudio.workers import export, export_generic, train, train_mlx, train_peft


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FTS_HOME", str(tmp_path))
    db.init()


def test_assistant_spans_and_labels_pure():
    inst, resp = "<U>", "<A>"
    text = "<U>q1</s><A>a1</s><U>q2</s><A>a2</s>"
    spans = masking.assistant_spans(text, inst, resp)
    assert [text[s:e] for s, e in spans] == ["a1</s>", "a2</s>"]                  # includes the end-of-turn token
    offs = [(0, 3), (3, 5), (5, 9), (9, 12), (12, 14), (14, 18)]
    assert masking.mask_labels(list(range(6)), offs, [(12, 18)]) == [-100, -100, -100, -100, 4, 5]
    assert masking.assistant_spans("no markers", inst, resp) == []


def test_tokenize_example_with_a_real_fast_tokenizer(tmp_path):
    pytest.importorskip("transformers")
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["[UNK]", "<|im_start|>", "<|im_end|>", "user", "assistant", "hi", "there", "two", "\n", " "]
    t = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Split(Regex(r"<\|im_start\|>|<\|im_end\|>|\w+|\n"), "isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")
    tok.chat_template = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "there"},
            {"role": "user", "content": "hi two"}, {"role": "assistant", "content": "there two"}]
    ex = masking.tokenize_example(tok, msgs, "<|im_start|>user\n", "<|im_start|>assistant\n", 100)
    trained = [words[i] for i in ex["labels"] if i != -100]
    assert trained == ["there", "<|im_end|>", "\n", "there", " ", "two", "<|im_end|>", "\n"]       # both replies, nothing else
    assert masking.tokenize_example(tok, msgs, "<|im_start|>user\n", "<|im_start|>assistant\n", 5) is None   # too long: dropped
    assert masking.tokenize_example(tok, msgs[:1], "<|im_start|>user\n", "<|im_start|>assistant\n", 100) is None  # no reply


def test_dtype_choice_and_trainingarguments_rename():
    assert train_peft.pick_dtype("auto", "cuda") == "bf16" and train_peft.pick_dtype("auto", "mps") == "fp32"
    assert train_peft.pick_dtype("auto", "cpu") == "fp32" and train_peft.pick_dtype("fp16", "mps") == "fp16"
    with pytest.raises(ValueError):
        train_peft.pick_dtype("half", "cuda")

    class Old:
        def __init__(self, output_dir=None, evaluation_strategy=None):
            self.v = evaluation_strategy

    assert train_peft.make_args(Old, output_dir="x", eval_strategy="steps").v == "steps"


def mlx_ds(tmp_path, n=6, val=2):
    d = tmp_path / "ds"
    d.mkdir()
    rows = [json.dumps({"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}) for i in range(n)]
    (d / "train.jsonl").write_text("\n".join(rows) + "\n")
    (d / "val.jsonl").write_text("\n".join(rows[:val]) + "\n" if val else "")
    return {"train": d / "train.jsonl", "val": d / "val.jsonl", "report": {"sha256": "abc", "val": val}}


def test_mlx_command_parsing_and_run(tmp_path, monkeypatch):
    cfg = {"base_model": "org/model", "batch_size": 2, "epochs": 2, "max_steps": None, "learning_rate": 2e-4, "lora_r": 8}
    run = tmp_path / "run"
    run.mkdir()
    cmd = train_mlx.build_command(run, cfg, run / "d", 10)
    assert cmd[cmd.index("--iters") + 1] == "10" and "--mask-prompt" in cmd and cmd[cmd.index("--model") + 1] == "org/model"
    assert train_mlx.build_command(run, {**cfg, "max_steps": 7}, run / "d", 10)[cmd.index("--iters") + 1] == "7"
    assert train_mlx.parse_line("Iter 10: Train loss 2.345, Learning Rate 1e-05") == {"step": 10, "loss": 2.345}
    assert train_mlx.parse_line("Iter 20: Val loss 1.9, Val took 1s") == {"step": 20, "eval_loss": 1.9} and train_mlx.parse_line("noise") is None

    class FakeProc:
        returncode = 0
        stdout = iter(["Starting\n", "Iter 5: Train loss 3.0, lr\n", "Iter 10: Train loss 2.0, lr\n", "Iter 10: Val loss 2.5\n"])
        def wait(self):
            return 0
    monkeypatch.setattr(train_mlx.subprocess, "Popen", lambda *a, **k: FakeProc())
    res = train_mlx.run_mlx(run, cfg, mlx_ds(tmp_path), {"accelerator": None})
    assert res["backend"] == "mlx" and res["final_loss"] == 2.0 and res["eval_loss"] == 2.5 and res["loss_curve"] == [[5, 3.0], [10, 2.0]]
    assert (run / "mlx_data" / "train.jsonl").exists() and json.loads((run / "mlx_config.yaml").read_text())["lora_parameters"]["rank"] == 8
    assert json.loads((run / "result.json").read_text())["backend"] == "mlx"


@pytest.mark.parametrize("engine,module,fn", [("peft", train_peft, "run_peft"), ("mlx", train_mlx, "run_mlx")])
def test_train_dispatches_to_the_chosen_engine_without_importing_unsloth(tmp_path, monkeypatch, engine, module, fn):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "report.json").write_text(json.dumps({"val": 0, "sha256": "s", "tokenizer": None,
                                               "template_probe": {"instruction_part": "<u>", "response_part": "<a>"}}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "params.json").write_text(json.dumps({"dataset": {"dir": str(d)}, "base_model": "org/model", "backend": engine}))
    monkeypatch.setattr(train, "require_accelerator", lambda **k: {"accelerator": "rocm", "vendor": "amd", "status": "ok"})
    monkeypatch.setattr(train, "choose_backend", lambda req, rep, **k: req)
    calls = []
    monkeypatch.setattr(module, fn, lambda *a, **k: calls.append(a))
    monkeypatch.delitem(sys.modules, "unsloth", raising=False)
    train.main(str(run))
    assert len(calls) == 1 and "unsloth" not in sys.modules


def test_export_generic_builds_the_llama_cpp_pipeline(tmp_path, monkeypatch):
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("#")
    monkeypatch.setenv("FTS_LLAMA_CPP", str(llama))
    monkeypatch.setenv("FTS_LLAMA_QUANTIZE", "/bin/llama-quantize")
    run, adapter = tmp_path / "run", tmp_path / "t1" / "adapter"
    run.mkdir()
    adapter.mkdir(parents=True)
    seen = []

    def fake_run(cmd):
        cmd = [str(c) for c in cmd]
        seen.append(cmd)
        out = cmd[cmd.index("--outfile") + 1] if "--outfile" in cmd else cmd[2]
        Path(out).write_text("gguf")

    merged = []
    files = export_generic.export_generic(run, {}, {"base_model": "org/model"}, adapter, ["q4_k_m", "q8_0"], "peft",
                                          merge=lambda b, a, o: merged.append((b, str(o))), run_cmd=fake_run)
    assert merged == [("org/model", str(run / "merged"))]
    assert "convert_hf_to_gguf.py" in seen[0][1] and seen[0][-2:] == ["--outtype", "f16"]
    assert [c[3] for c in seen[1:]] == ["Q4_K_M", "Q8_0"] and seen[1][0] == "/bin/llama-quantize"
    assert sorted(f.name for f in files) == ["t1-Q4_K_M.gguf", "t1-Q8_0.gguf"]               # the f16 intermediate is removed
    keep = export_generic.export_generic(run, {}, {"base_model": "m"}, adapter, ["f16"], "peft", merge=lambda *a: None, run_cmd=fake_run)
    assert "t1-f16.gguf" in [f.name for f in keep]                                           # asked for f16: it is kept
    monkeypatch.delenv("FTS_LLAMA_CPP")
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="FTS_LLAMA_CPP"):
        export_generic.find_converter()
    with pytest.raises(ValueError, match="base_model"):
        export_generic.export_generic(run, {}, {}, adapter, ["q4_k_m"], "peft")


def test_export_main_routes_non_unsloth_backends_and_queues_eval(tmp_path, monkeypatch):
    adapter = tmp_path / "trainjob" / "adapter"
    adapter.mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    (run / "params.json").write_text(json.dumps({"adapter_dir": str(adapter), "base_model": "org/m", "backend": "mlx",
                                                 "quantizations": ["q4_k_m"], "auto_eval": {"custom": {"file": "t.jsonl"}}}))

    def fake_export(run_, p, cfg, adapter_, quants, backend, **k):
        (run_ / "gguf").mkdir(exist_ok=True)
        (run_ / "gguf" / "x-Q4_K_M.gguf").write_text("gguf")
    monkeypatch.setattr(export_generic, "export_generic", fake_export)
    monkeypatch.delitem(sys.modules, "unsloth", raising=False)
    export.main(str(run))
    res = json.loads((run / "result.json").read_text())
    assert res["backend"] == "mlx" and res["ggufs"][0]["quant"] == "q4_k_m" and len(res["eval_jobs"]) == 1
    assert db.get_job(res["eval_jobs"][0])["kind"] == "eval" and "unsloth" not in sys.modules
