"""Stage 2: distill a LoRA teacher into PQ2_0 block scales; ternary codes never change.

  teacher  native teacher candidate logits for every tokenized training prompt
  canary   whole-GPU parity, backward, memory and export check before training
  train    learn one factor per 128-weight block, select on development NLL, and
           export a plain PQ2_0 GGUF without an adapter

Canary and training run only on an operator-authorized GPU with the adapter rails.
The (factor - 1)^2 penalty is a mean over tensors of per-tensor means, so under
AdamW it is numerically inert; the effective bounds are the learning rate, the
step count and development early stopping. The default is kept for comparability.
"""
import argparse
import gc
import hashlib
import json
import math
import os
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
from shingi.pq2_scales import BLOCK, export_scaled, scaled_bytes
from shingi.ternary_training import BASE_SHA256, GLOBAL, MAPPING, decode_pq2, hadamard, load_bonsai, reorder_rows
from train_decision_adapter import rows, sha, validate_fitting_sources

TEACHER_WEIGHT, REGULARIZATION, LEARNING_RATE, ACCUMULATION, WARMUP = .7, 1e-3, 2e-4, 8, 32
CANARY_CONTEXT = 2048
CHUNK = 1 << 28  # FP32 elements expanded at once; only the 1.27B-element output matrix is split
NATIVE_GATE = {"argmax_agreement": .95, "mean_tvd": .03, "max_tvd": .10, "centered_logit_rmse": .30}


def teacher(args):
    """Candidate logits from the stage 1 native adapter, before any scale training."""
    data = args.data
    out = args.output
    if out.exists():
        raise SystemExit("preserve existing teacher logits")
    if sha(args.model) != BASE_SHA256:
        raise RuntimeError("base checksum mismatch")
    readout = NativeReadout("artifacts/bin/readout", args.model, 4096, adapter=args.adapter)
    try:
        with out.open("x") as stream:
            for row in rows(data / "tokenized.jsonl"):
                logits = readout.infer(row["prompt"], row["labels"])["logits"]
                stream.write(json.dumps({"id": row["id"], "logits": logits}) + "\n")
    finally:
        readout.close()
    receipt = {"model_sha256": BASE_SHA256, "adapter_sha256": sha(args.adapter),
               "tokens_sha256": sha(data / "tokenized.jsonl"),
               "teacher_sha256": sha(out), "executable_sha256": sha("artifacts/bin/readout")}
    out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


def target_map(tensors, fields):
    """(name, type, rows) GGUF tensors -> {name: (module path, row permutation)} of scaled matmuls."""
    g = lambda k: fields["qwen35." + k]
    nk, nv = int(g("ssm.group_count")), int(g("ssm.time_step_rank"))
    hk, hv = int(g("ssm.state_size")), int(g("ssm.inner_size")) // nv
    embeddings = set(fields["prism.hadamard.inverse_weight_names"]) | {"token_embd.weight"}
    targets = {}
    for name, kind, rows_ in tensors:
        if kind != "PQ2_0" or name in embeddings:
            continue
        if name in GLOBAL:
            path, stem = GLOBAL[name], name
        else:
            _, layer, stem = name.split(".", 2)
            path = f"model.layers.{layer}." + MAPPING[stem]
        # Loaded row i came from stored row perm[i] (see load_bonsai).
        targets[name] = (path.rsplit(".", 1)[0], reorder_rows(np.arange(rows_), stem, nk, nv, hk, hv))
    return targets


def gguf(prism, path):
    sys.path.insert(0, str(Path(prism) / "gguf-py"))
    from gguf import GGUFReader
    return GGUFReader(str(path))


def pq2_targets(model_path, prism):
    """GGUF PQ2_0 tensor name -> (module path, row permutation) for every scaled matmul."""
    reader = gguf(prism, model_path)
    f = {k: v.contents() for k, v in reader.fields.items() if not k.startswith("tokenizer.")}
    tensors = [(t.name, t.tensor_type.name, int(t.shape[1])) for t in reader.tensors if t.tensor_type.name == "PQ2_0"]
    return target_map(tensors, f), int(f["prism.hadamard.block_size"])


def spans(rows_, width, chunk):
    step = max(1, chunk // width)
    return [slice(s, min(s + step, rows_)) for s in range(0, rows_, step)]


def scaled_matmul(chunk=CHUNK):
    import torch

    class ScaledMatmul(torch.autograd.Function):
        # y = x (W * f)^T with one factor per 128-weight block. Only f is trainable;
        # the expanded FP32 matrix is transient and built in bounded row chunks.
        @staticmethod
        def forward(ctx, x, weight, factors):
            ctx.save_for_backward(x, weight, factors)
            x = x.float()
            out = [torch.nn.functional.linear(x, weight[s].float() * factors[s].repeat_interleave(BLOCK, dim=1))
                   for s in spans(*weight.shape, chunk)]
            return out[0] if len(out) == 1 else torch.cat(out, -1)

        @staticmethod
        def backward(ctx, grad):
            x, weight, factors = ctx.saved_tensors
            flat_x = x.float().reshape(-1, x.shape[-1])
            flat_g = grad.float().reshape(-1, grad.shape[-1])
            grad_x = torch.zeros_like(flat_x)
            grad_f = torch.empty_like(factors)
            for s in spans(*weight.shape, chunk):
                w = weight[s].float()
                grad_x += flat_g[:, s] @ (w * factors[s].repeat_interleave(BLOCK, dim=1))
                grad_f[s] = ((flat_g[:, s].T @ flat_x) * w).reshape(w.shape[0], -1, BLOCK).sum(-1)
            return grad_x.reshape(x.shape), None, grad_f

    return ScaledMatmul


def attach_scales(model, targets, block):
    import torch
    from torch import nn
    ScaledMatmul = scaled_matmul()

    class Scaled(nn.Module):
        def __init__(self, inner):
            super().__init__()
            if isinstance(inner, Scaled) or getattr(inner, "bias", None) is not None:
                raise ValueError("scaled target must be an unscaled bias-free matmul")
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


def check_wrapped(expected, targets, wrapped, shapes, factors):
    """Every PQ2_0 non-embedding tensor is wrapped exactly once and maps back to its stored shape.

    expected: GGUF tensor names; wrapped: module paths holding a scale wrapper;
    shapes: name -> stored (rows, width); factors: name -> stored-order factors.
    """
    if set(targets) != set(expected):
        raise ValueError("scaled targets differ from the PQ2_0 matmul tensors")
    paths = [path for path, _ in targets.values()]
    if len(set(paths)) != len(paths) or sorted(wrapped) != sorted(paths):
        raise ValueError("every target must be wrapped exactly once")
    for name, (rows_, width) in shapes.items():
        if factors[name].shape != (rows_, width // BLOCK) or sorted(targets[name][1]) != list(range(rows_)):
            raise ValueError(f"{name} factors do not map back to the stored shape")
    return {"scaled_tensors": len(paths), "factors": int(sum(f.size for f in factors.values()))}


def unit_parity(reference, actual):
    """Unit factors must reproduce the unscaled forward up to FP32 rounding."""
    from training_canary import compare
    result = compare(actual, reference)
    result["max_abs_difference"] = float(max(np.abs(np.subtract(a, b)).max() for a, b in zip(actual, reference)))
    result["passed"] = result["centered_logit_rmse"] <= 1e-3 and result["argmax_agreement"] == 1.
    return result


def native_parity(trained, native):
    """Stage 1 adapter-export tolerance: native kernels, F16 KV and FP16 scale rounding differ from FP32."""
    from training_canary import compare
    result = compare(trained, native)
    result["passed"] = (result["argmax_agreement"] >= NATIVE_GATE["argmax_agreement"]
                        and all(result[k] <= v for k, v in NATIVE_GATE.items() if k != "argmax_agreement"))
    return result


def differing_bytes(a, b, chunk=1 << 28):
    count = 0
    with open(a, "rb") as x, open(b, "rb") as y:
        while True:
            p, q = x.read(chunk), y.read(chunk)
            if len(p) != len(q):
                raise ValueError("file sizes differ")
            if not p:
                return count
            count += int(np.count_nonzero(np.frombuffer(p, np.uint8) != np.frombuffer(q, np.uint8)))


def verify_export(base, output, factors, prism):
    """The export equals the base except the expected scale bytes at every named tensor's offset."""
    before = {t.name: t for t in gguf(prism, base).tensors}
    after = {t.name: t for t in gguf(prism, output).tensors}
    changed = 0
    for name, value in factors.items():
        b, a = before[name], after[name]
        width, rows_ = (int(n) for n in b.shape[:2])
        old, new = np.asarray(b.data).reshape(-1), np.asarray(a.data).reshape(-1)
        if int(a.data_offset) != int(b.data_offset) or not np.array_equal(new, scaled_bytes(old, rows_, width, value)):
            raise ValueError(f"{name} export differs from its expected scale bytes")
        changed += int(np.count_nonzero(new != old))
    if differing_bytes(base, output) != changed:
        raise ValueError("bytes outside the scaled tensors changed")
    return {"tensors": len(factors), "changed_bytes": changed, "bytes": Path(output).stat().st_size}


def distill_loss(logits, teacher_logits, target, weight):
    import torch
    mixed = weight * torch.softmax(teacher_logits, -1) + (1 - weight) * target
    return -(mixed * torch.log_softmax(logits, -1)).sum()


def penalty(params, regularization):
    return regularization * sum(((p - 1) ** 2).mean() for p in params) / len(params)


def memory(rails):
    import torch
    torch.cuda.synchronize()
    rails.assert_post_first_backward_free()
    return {"free_gib": torch.cuda.mem_get_info()[0] / 2**30, "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}


def canary_torch(args, prompts, result, save):
    """Differentiable half of the canary. Returns only CPU values so GPU memory can be released."""
    import torch
    from shingi import gpu as rails
    rails.require_training_gpu()
    torch.manual_seed(20260925)
    model, info = load_bonsai(args.model, args.prism)
    result.update(loader=info)
    result["memory"] = {"after_load": memory(rails)}

    def forward(ids, candidates):
        h = model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).last_hidden_state[:, -1:, :]
        return model.lm_head(h)[0, 0, candidates].float()

    def logits():
        with torch.no_grad():
            return [forward(p["input_ids"], p["candidate_ids"]).cpu().tolist() for p in prompts]
    reference = logits()
    save()
    # 2. Wrap every PQ2_0 matmul; check coverage, stored shapes and the loaded-row mapping.
    reader = gguf(args.prism, args.model)
    f = {k: v.contents() for k, v in reader.fields.items() if not k.startswith("tokenizer.")}
    embeddings = set(f["prism.hadamard.inverse_weight_names"]) | {"token_embd.weight"}
    stored = {t.name: t for t in reader.tensors if t.tensor_type.name == "PQ2_0" and t.name not in embeddings}
    targets, block = pq2_targets(args.model, args.prism)
    modules = attach_scales(model, targets, block)
    Scaled = type(next(iter(modules.values()))[0])
    wrapped = [n for n, m in model.named_modules() if isinstance(m, Scaled)]
    shapes = {n: (int(t.shape[1]), int(t.shape[0])) for n, t in stored.items()}
    result["wrapping"] = check_wrapped(stored, targets, wrapped, shapes, stored_factors(modules))
    mapped = 0
    for name, (module, perm) in modules.items():
        rows_, width = shapes[name]
        moved = np.flatnonzero(perm != np.arange(rows_))
        for i in {0, rows_ - 1, *moved[:2].tolist(), *moved[-2:].tolist()}:
            row = decode_pq2(np.asarray(stored[name].data)[perm[i]:perm[i] + 1], 1, width)[0]
            if not np.array_equal(module.inner.weight[i].cpu().numpy(), row):
                raise ValueError(f"{name} loaded row {i} is not stored row {perm[i]}")
            mapped += 1
    result["wrapping"]["row_mapping_rows_checked"] = mapped
    params = [m.factors for m, _ in modules.values()]
    result["memory"]["after_attach"] = memory(rails)
    save()
    # 3. Unit factors reproduce the unscaled forward.
    result["unit_parity"] = unit_parity(reference, logits())
    save()
    if not result["unit_parity"]["passed"]:
        raise RuntimeError("unit-factor parity failed")
    # 4. One teacher-style step on a repeated 2,048-token prompt (synthetic; never reused).
    first = prompts[0]
    ids = (first["input_ids"] * (CANARY_CONTEXT // len(first["input_ids"]) + 1))[:CANARY_CONTEXT]
    result["backward_context_tokens"] = len(ids)
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, betas=(.9, .95), weight_decay=0.)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    started = time.monotonic()
    target = torch.zeros(len(first["candidate_ids"]), device="cuda")
    target[1] = 1
    loss = distill_loss(forward(ids, first["candidate_ids"]), torch.tensor(reference[0], device="cuda"), target, TEACHER_WEIGHT)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite canary loss")
    loss.backward()
    result["memory"]["first_backward"] = memory(rails)
    penalty(params, REGULARIZATION).backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
    result.update(loss=loss.item(), gradient_norm=norm.item())
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise RuntimeError("canary factor gradient must be finite and non-zero")
    before = [p.detach().clone() for p in params]
    optimizer.step()
    result["changed_factors"] = int(sum((p.detach() != b).sum().item() for p, b in zip(params, before)))
    del before
    result["step_seconds"] = time.monotonic() - started
    result["memory"]["after_step"] = memory(rails)
    save()
    if not result["changed_factors"]:
        raise RuntimeError("optimizer step did not change any factor")
    model.eval()
    model.gradient_checkpointing_disable()
    trained = logits()
    stepped = stored_factors(modules)
    with torch.no_grad():
        for p in params:
            p.fill_(1.)
    unit = stored_factors(modules)
    if not all((v == 1).all() for v in unit.values()):
        raise RuntimeError("factor reset failed")
    return trained, stepped, unit


def canary(args):
    """Whole-GPU stage 2 canary; parity.json is written in every case and ends with passed."""
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"passed": False, "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "native_kv": "f16", "learning_rate": LEARNING_RATE, "teacher_weight": TEACHER_WEIGHT,
              "native_gate": NATIVE_GATE}
    save = lambda: (args.output / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    try:
        from shingi.gpu import selected_gpu
        from training_canary import make_prompts
        selected_gpu()
        # 1. Pinned base and prompts tokenized exactly as the native runtime does.
        if sha(args.model) != BASE_SHA256:
            raise RuntimeError("base checksum mismatch")
        result["model_sha256"] = BASE_SHA256
        os.environ["SHINGI_KV_F16"] = "1"
        prompts = make_prompts()
        tokenizer = NativeTokenizer("artifacts/bin/readout", args.model)
        try:
            for p in prompts:
                p.update(tokenizer.encode(p["prompt"], p["labels"]))
        finally:
            tokenizer.close()
        trained, stepped, unit = canary_torch(args, prompts, result, save)
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        # 5. Unit export is byte-identical; a stepped export changes only scale bytes and loads natively.
        path = args.output / "unit.gguf"
        export_scaled(args.model, path, unit, args.prism)
        result["unit_export_sha256"] = sha(path)
        path.unlink()
        if result["unit_export_sha256"] != BASE_SHA256:
            raise RuntimeError("unit-factor export is not byte-identical to the base")
        path = args.output / "canary.gguf"
        export_scaled(args.model, path, stepped, args.prism)
        try:
            result["canary_export"] = verify_export(args.model, path, stepped, args.prism)
            save()
            if not result["canary_export"]["changed_bytes"]:
                raise RuntimeError("stepped export changed no scale bytes")
            native = NativeReadout("artifacts/bin/readout", path, 2048)
            try:
                reloaded = [native.infer(p["prompt"], p["labels"])["logits"] for p in prompts]
            finally:
                native.close()
        finally:
            path.unlink()
        result["native_parity"] = native_parity(trained, reloaded)
        if not result["native_parity"]["passed"]:
            raise RuntimeError("stepped export does not reproduce the differentiable logits natively")
        result["passed"] = True
    except BaseException as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        print(json.dumps({k: result.get(k) for k in ("passed", "wrapping", "unit_parity", "native_parity", "failure")}), flush=True)


def train(args):
    import torch
    from safetensors.torch import save_file
    from safetensors.numpy import load_file
    from shingi import gpu as rails
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("source must be committed")
    token_manifest = json.loads((args.data / "tokenized-manifest.json").read_text())
    parity = json.loads(args.canary.read_text())
    if (not parity.get("passed") or parity.get("model_sha256") != BASE_SHA256
            or parity.get("backward_context_tokens") != token_manifest["max_tokens"]):
        raise RuntimeError("successful matching stage 2 canary required")
    if sha(args.model) != BASE_SHA256:
        raise RuntimeError("base checksum mismatch")
    receipt = json.loads(args.teacher.with_suffix(".json").read_text())
    if (receipt.get("model_sha256") != BASE_SHA256 or receipt["teacher_sha256"] != sha(args.teacher)
            or receipt["tokens_sha256"] != sha(args.data / "tokenized.jsonl")):
        raise RuntimeError("teacher logits do not match the base and tokenized training data")
    manifest = json.loads((args.data / "manifest.json").read_text())
    for split in ("train-prompts", "dev"):
        if sha(args.data / (split + ".jsonl")) != manifest[split + "_sha256"]:
            raise RuntimeError("dataset hash mismatch")
    if token_manifest["data_manifest_sha256"] != sha(args.data / "manifest.json") or token_manifest["tokens_sha256"] != receipt["tokens_sha256"]:
        raise RuntimeError("prepared tokenization provenance differs")
    prepared = rows(args.data / "tokenized.jsonl")
    teacher_logits = {r["id"]: r["logits"] for r in rows(args.teacher)}
    dev = rows(args.data / "dev.jsonl")
    validate_fitting_sources(manifest, prepared, dev)
    if set(teacher_logits) != {r["id"] for r in prepared} or any(len(teacher_logits[r["id"]]) != len(r["candidate_ids"]) for r in prepared):
        raise RuntimeError("teacher logits must cover every training prompt")
    args.output.mkdir(parents=True, exist_ok=False)
    stop = None

    def request_stop(*unused):
        nonlocal stop
        stop = "signal"
    signal.signal(signal.SIGTERM, request_stop)
    rails.require_training_gpu()
    torch.set_num_threads(12)
    torch.manual_seed(20260925)
    model, info = load_bonsai(args.model, args.prism)
    targets, block = pq2_targets(args.model, args.prism)
    modules = attach_scales(model, targets, block)
    params = [m.factors for m, _ in modules.values()]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(.9, .95), weight_decay=0.)
    tokenizer = NativeTokenizer("artifacts/bin/readout", args.model)
    updates = math.ceil(len(prepared) / ACCUMULATION)
    started = time.monotonic()
    run = {"code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "model_sha256": BASE_SHA256, "model_bytes": args.model.stat().st_size, "adapter_sha256": receipt["adapter_sha256"],
           "canary_sha256": sha(args.canary), "teacher": receipt, "teacher_file": str(args.teacher),
           "dataset_manifest_sha256": sha(args.data / "manifest.json"), "tokens_sha256": receipt["tokens_sha256"],
           "profile": manifest.get("profile"), "prepared_prompts": len(prepared), "dev_records": len(dev),
           "loader": info, "scaled_tensors": len(modules), "trainable_factors": sum(p.numel() for p in params),
           "hyperparameters": {"teacher_weight": args.teacher_weight, "regularization": args.regularization,
                               "learning_rate": args.learning_rate, "gradient_accumulation": ACCUMULATION,
                               "warmup_updates": WARMUP, "schedule": "linear warmup, cosine to 0.2 of peak",
                               "optimizer": "AdamW betas (0.9, 0.95), no weight decay", "gradient_clip": 1.},
           "loss": "cross entropy against teacher_weight x teacher softmax + (1 - teacher_weight) x label target",
           "checkpoint_selection": "lowest uncalibrated development NLL; test and calibration not read",
           "early_stop_patience": 2, "planned_updates": updates, "max_seconds": args.max_seconds,
           "eval_every": args.eval_every, "checkpoints": [], "status": "running"}
    save_run = lambda: (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    class Backend:
        def infer(self, prompt, labels):
            r = tokenizer.encode(prompt, labels)
            if r["input_tokens"] > 8192:
                raise ValueError("development prompt exceeds 8192-token inference bound")
            ids = torch.tensor([r["input_ids"]], device="cuda")
            h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1:, :]
            out = model.lm_head(h)[0, 0, r["candidate_ids"]].float().cpu().tolist()
            rails.assert_post_first_backward_free()
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
        with (args.output / f"dev-predictions-{step:05}.json").open("x") as stream:
            json.dump(predictions, stream)
        metrics = report(dev, predictions)
        (args.output / f"dev-{step:05}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        nll = metrics["overall"]["nll"]["mean"]
        if not math.isfinite(nll) or metrics["overall"]["valid"] != len(dev):
            raise RuntimeError("invalid development evaluation")
        path = args.output / f"factors-{step:05}.safetensors"
        save_file({name: torch.from_numpy(v) for name, v in stored_factors(modules).items()}, str(path))
        record = {"update": step, "dev_nll": nll, "dev_accuracy": metrics["overall"]["accuracy_failures_incorrect"],
                  "factors": str(path), "sha256": sha(path), "elapsed_seconds": time.monotonic() - started,
                  "memory": memory(rails)}
        run["checkpoints"].append(record)
        if best is None or nll < best["dev_nll"]:
            best = record
        run["selected"] = best
        save_run()
        print(json.dumps(record), flush=True)
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    try:
        evaluate(0)
        stale = completed = last_eval = 0
        with (args.output / "steps.jsonl").open("x") as log:
            for step in range(updates):
                if stop is None and time.monotonic() - started >= args.max_seconds:
                    stop = "max-seconds"
                if stop:
                    break
                group = prepared[step * ACCUMULATION:(step + 1) * ACCUMULATION]
                lr = args.learning_rate * min(1., (step + 1) / WARMUP) * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / max(1, updates - 1))))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                before = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                total = 0.
                for item in group:
                    ids = torch.tensor([item["input_ids"]], device="cuda")
                    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1:, :]
                    logits = model.lm_head(h)[0, 0, item["candidate_ids"]].float()
                    loss = distill_loss(logits, torch.tensor(teacher_logits[item["id"]], device="cuda"),
                                        torch.tensor(item["target"], device="cuda"), args.teacher_weight) / len(group)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite loss")
                    loss.backward()
                    total += loss.item()
                    if "first_backward" not in run:
                        run["first_backward"] = memory(rails)
                        save_run()
                reg = penalty(params, args.regularization)
                reg.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError("non-finite gradient")
                optimizer.step()
                completed = step + 1
                log.write(json.dumps({"update": completed, "loss": total, "penalty": reg.item(), "learning_rate": lr,
                                      "gradient_norm": norm.item(), "seconds": time.monotonic() - before,
                                      "memory": memory(rails)}) + "\n")
                log.flush()
                run["completed_updates"] = completed
                if completed % args.eval_every == 0:
                    previous = best["dev_nll"]
                    evaluate(completed)
                    last_eval = completed
                    stale = stale + 1 if best["dev_nll"] >= previous else 0
                    if stale >= 2:
                        stop = "two development evaluations without improvement"
                        break
        # A signal leaves no time for another evaluation; export the best evaluated factors.
        if stop != "signal" and completed != last_eval:
            evaluate(completed)
        run.update(stop_reason=stop or "completed", status="finished" if completed == updates else "bounded-stop",
                   improved=best["update"] > 0 and best["dev_nll"] < run["checkpoints"][0]["dev_nll"])
        save_run()
    except BaseException as exc:
        run.update(status="failed", failure=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        save_run()
        raise
    finally:
        tokenizer.close()
    del model, modules, params, optimizer, engine
    gc.collect()
    torch.cuda.empty_cache()
    output = args.output / "shingi-ternary.gguf"
    factors = load_file(best["factors"])
    export_scaled(args.model, output, factors, args.prism)
    run.update(export=str(output), export_sha256=sha(output), export_bytes=output.stat().st_size,
               export_check=verify_export(args.model, output, factors, args.prism), elapsed_seconds=time.monotonic() - started)
    save_run()
    if run["export_bytes"] != run["model_bytes"]:
        raise RuntimeError("export size differs from the base")
    print(json.dumps(run["selected"]))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--model", type=Path, required=True)
    t.add_argument("--adapter", type=Path, required=True)
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("canary")
    c.add_argument("--model", type=Path, required=True)
    c.add_argument("--prism", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("train")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--prism", type=Path, required=True)
    r.add_argument("--canary", type=Path, required=True)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--teacher", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--teacher-weight", type=float, default=TEACHER_WEIGHT)
    r.add_argument("--regularization", type=float, default=REGULARIZATION)
    r.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    r.add_argument("--eval-every", type=int, default=256)
    r.add_argument("--max-seconds", type=int, default=86400)
    args = parser.parse_args()
    {"teacher": teacher, "canary": canary, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
