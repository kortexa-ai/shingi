"""Stage 2: distill a LoRA teacher into PQ2_0 block scales; ternary codes never change.

  teacher  native teacher candidate logits for every tokenized training prompt
  train    learn one factor per 128-weight block, select on development NLL, and
           export a plain PQ2_0 GGUF without an adapter

Training runs only on an operator-authorized GPU with the same rails as adapter training.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from shingi.backend import NativeReadout
from shingi.decision import DecisionEngine
from shingi.metrics import report
from shingi.native_tokenizer import NativeTokenizer
from shingi.pq2_scales import BLOCK, export_scaled
from shingi.ternary_training import BASE_SHA256, GLOBAL, MAPPING, hadamard, load_bonsai, reorder_rows
from train_decision_adapter import rows, sha


def teacher(args):
    """Candidate logits from the stage 1 native adapter, before any scale training."""
    data = args.data
    out = args.output
    if out.exists():
        raise SystemExit("preserve existing teacher logits")
    readout = NativeReadout("artifacts/bin/readout", args.model, 4096, adapter=args.adapter)
    try:
        with out.open("x") as stream:
            for row in rows(data / "tokenized.jsonl"):
                logits = readout.infer(row["prompt"], row["labels"])["logits"]
                stream.write(json.dumps({"id": row["id"], "logits": logits}) + "\n")
    finally:
        readout.close()
    receipt = {"adapter_sha256": sha(args.adapter), "tokens_sha256": sha(data / "tokenized.jsonl"),
               "teacher_sha256": sha(out), "executable_sha256": sha("artifacts/bin/readout")}
    out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


def pq2_targets(model_path, prism):
    """GGUF PQ2_0 tensor name -> (module path, row permutation) for every scaled matmul."""
    sys.path.insert(0, str(Path(prism) / "gguf-py"))
    from gguf import GGUFReader
    reader = GGUFReader(str(model_path))
    f = {k: v.contents() for k, v in reader.fields.items() if not k.startswith("tokenizer.")}
    g = lambda k: f["qwen35." + k]
    nk, nv = int(g("ssm.group_count")), int(g("ssm.time_step_rank"))
    hk, hv = int(g("ssm.state_size")), int(g("ssm.inner_size")) // nv
    embeddings = set(f["prism.hadamard.inverse_weight_names"]) | {"token_embd.weight"}
    targets = {}
    for tensor in reader.tensors:
        if tensor.tensor_type.name != "PQ2_0" or tensor.name in embeddings:
            continue
        if tensor.name in GLOBAL:
            path, stem = GLOBAL[tensor.name], tensor.name
        else:
            _, layer, stem = tensor.name.split(".", 2)
            path = f"model.layers.{layer}." + MAPPING[stem]
        rows_ = int(tensor.shape[1])
        perm = reorder_rows(np.arange(rows_), stem, nk, nv, hk, hv)
        targets[tensor.name] = (path.rsplit(".", 1)[0], perm)
    return targets, int(f["prism.hadamard.block_size"])


def scaled_matmul():
    import torch

    class ScaledMatmul(torch.autograd.Function):
        # y = x (W * f)^T with one factor per 128-weight block. Only f is trainable;
        # the expanded FP32 matrix is transient, as in the frozen LoRA path.
        @staticmethod
        def forward(ctx, x, weight, factors):
            ctx.save_for_backward(x, weight, factors)
            effective = weight.float() * factors.repeat_interleave(BLOCK, dim=1)
            return torch.nn.functional.linear(x.float(), effective)

        @staticmethod
        def backward(ctx, grad):
            x, weight, factors = ctx.saved_tensors
            effective = weight.float() * factors.repeat_interleave(BLOCK, dim=1)
            grad_x = torch.matmul(grad.float(), effective)
            grad_w = grad.float().reshape(-1, grad.shape[-1]).T @ x.float().reshape(-1, x.shape[-1])
            grad_f = (grad_w * weight.float()).reshape(weight.shape[0], -1, BLOCK).sum(-1)
            return grad_x, None, grad_f

    return ScaledMatmul


def attach_scales(model, targets, block):
    import torch
    from torch import nn
    ScaledMatmul = scaled_matmul()

    class Scaled(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            rows_, width = inner.weight.shape
            self.factors = nn.Parameter(torch.ones(rows_, width // BLOCK, device=inner.weight.device, dtype=torch.float32))

        def forward(self, x):
            if hasattr(self.inner, "signs"):  # Hadamard-folded stored weight
                x = hadamard(x, self.inner.signs, block)
            return ScaledMatmul.apply(x, self.inner.weight, self.factors)

    modules = {}
    for name, (path, perm) in targets.items():
        parent, _, attr = path.rpartition(".")
        module = Scaled(model.get_submodule(path))
        setattr(model.get_submodule(parent), attr, module)
        modules[name] = (module, perm)
    return modules


def stored_factors(modules):
    # Loaded row i came from stored row perm[i]; stored order is needed for export.
    out = {}
    for name, (module, perm) in modules.items():
        factors = module.factors.detach().cpu().numpy()
        stored = np.empty_like(factors)
        stored[perm] = factors
        out[name] = stored
    return out


def train(args):
    import torch
    from safetensors.torch import save_file
    from shingi import gpu as rails
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("source must be committed")
    if sha(args.model) != BASE_SHA256:
        raise RuntimeError("base checksum mismatch")
    receipt = json.loads(args.teacher.with_suffix(".json").read_text())
    if receipt["teacher_sha256"] != sha(args.teacher) or receipt["tokens_sha256"] != sha(args.data / "tokenized.jsonl"):
        raise RuntimeError("teacher logits do not match the tokenized training data")
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = rows(args.data / "tokenized.jsonl")
    teacher_logits = {r["id"]: r["logits"] for r in rows(args.teacher)}
    dev = rows(args.data / "dev.jsonl")
    rails.require_training_gpu()
    torch.manual_seed(20260925)
    model, info = load_bonsai(args.model, args.prism)
    targets, block = pq2_targets(args.model, args.prism)
    modules = attach_scales(model, targets, block)
    params = [m.factors for m, _ in modules.values()]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(.9, .95), weight_decay=0.)
    tokenizer = NativeTokenizer("artifacts/bin/readout", args.model)
    accumulation = 8
    updates = math.ceil(len(prepared) / accumulation)
    started = time.monotonic()
    stop = False

    def request_stop(*unused):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    run = {"code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "model_sha256": BASE_SHA256, "teacher": receipt, "loader": info, "scaled_tensors": len(modules),
           "trainable_factors": sum(p.numel() for p in params), "teacher_weight": args.teacher_weight,
           "regularization": args.regularization, "learning_rate": args.learning_rate,
           "planned_updates": updates, "max_seconds": args.max_seconds, "checkpoints": []}

    class Backend:
        def infer(self, prompt, labels):
            r = tokenizer.encode(prompt, labels)
            ids = torch.tensor([r["input_ids"]], device="cuda")
            h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1:, :]
            out = model.lm_head(h)[0, 0, r["candidate_ids"]].float().cpu().tolist()
            return {"logits": out, "input_tokens": r["input_tokens"], "candidate_ids": r["candidate_ids"],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    engine = DecisionEngine(Backend(), canonical_choices=True)
    best = None

    def evaluate(step):
        nonlocal best
        model.eval()
        model.gradient_checkpointing_disable()
        with torch.no_grad():
            predictions = {}
            for r in dev:
                answer, traces = engine.answer(r["state"], r["question"])
                predictions[r["id"]] = {"answer": answer, "traces": traces}
        metrics = report(dev, predictions)
        nll = metrics["overall"]["nll"]["mean"]
        path = args.output / f"factors-{step:05}.safetensors"
        save_file({name: torch.from_numpy(v) for name, v in stored_factors(modules).items()}, str(path))
        record = {"update": step, "dev_nll": nll, "dev_accuracy": metrics["overall"]["accuracy_failures_incorrect"],
                  "factors": str(path), "sha256": sha(path), "elapsed_seconds": time.monotonic() - started}
        run["checkpoints"].append(record)
        if best is None or nll < best["dev_nll"]:
            best = record
        (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    evaluate(0)
    stale = 0
    with (args.output / "steps.jsonl").open("x") as log:
        for step in range(updates):
            if stop or time.monotonic() - started >= args.max_seconds:
                break
            group = prepared[step * accumulation:(step + 1) * accumulation]
            lr = args.learning_rate * min(1., (step + 1) / 32) * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / max(1, updates - 1))))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for item in group:
                ids = torch.tensor([item["input_ids"]], device="cuda")
                h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1:, :]
                logits = model.lm_head(h)[0, 0, item["candidate_ids"]].float()
                soft = torch.softmax(torch.tensor(teacher_logits[item["id"]], device="cuda"), -1)
                target = args.teacher_weight * soft + (1 - args.teacher_weight) * torch.tensor(item["target"], device="cuda")
                loss = -(target * torch.log_softmax(logits, -1)).sum() / len(group)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
                loss.backward()
                total += loss.item()
            penalty = args.regularization * sum(((p - 1) ** 2).mean() for p in params) / len(params)
            penalty.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("non-finite gradient")
            optimizer.step()
            torch.cuda.synchronize()
            rails.assert_post_first_backward_free()
            log.write(json.dumps({"update": step + 1, "loss": total, "penalty": penalty.item(), "learning_rate": lr,
                                  "gradient_norm": norm.item(), "free_gib": torch.cuda.mem_get_info()[0] / 2**30}) + "\n")
            log.flush()
            if (step + 1) % args.eval_every == 0:
                previous = best["dev_nll"]
                evaluate(step + 1)
                stale = stale + 1 if best["dev_nll"] >= previous else 0
                if stale >= 2:
                    run["stop_reason"] = "two development evaluations without improvement"
                    break
    tokenizer.close()
    from safetensors.numpy import load_file
    del model
    torch.cuda.empty_cache()
    output = args.output / "shingi-ternary.gguf"
    export_scaled(args.model, output, load_file(best["factors"]), args.prism)
    run.update(selected=best, export=str(output), export_sha256=sha(output), elapsed_seconds=time.monotonic() - started)
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run["selected"]))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--model", type=Path, required=True)
    t.add_argument("--adapter", type=Path, required=True)
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("train")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--prism", type=Path, required=True)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--teacher", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--teacher-weight", type=float, default=.7)
    r.add_argument("--regularization", type=float, default=1e-3)
    r.add_argument("--learning-rate", type=float, default=2e-4)
    r.add_argument("--eval-every", type=int, default=256)
    r.add_argument("--max-seconds", type=int, default=86400)
    args = parser.parse_args()
    teacher(args) if args.command == "teacher" else train(args)


if __name__ == "__main__":
    main()
