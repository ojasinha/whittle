"""Config-driven, resumable trainer for Plan A / B1 / B2.

  python train/train.py configs/smoke_ettin.yaml [--resume] [--set train.max_steps=50 optim.lr_pre=3e-5]

Writes to cfg.out_dir:
  metrics.csv            step, split, loss, lr, tokens/s, mem, gates...
  ckpt/                  latest resumable checkpoint (model + optimizer + scheduler + rng + data position)
  best/                  best-val model (for eval/packaging)
  summary.json           final stats (wall time, best val, tokens seen)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "train"))

from data import DenoiseData, MixedData, PairData, token_budget_batches  # noqa: E402

from whittle.wrappers import build_wrapper, load_wrapper  # noqa: E402


def set_path(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def build_dataset(dcfg, seed):
    kind = dcfg["kind"]
    if kind == "pairs":
        return PairData(dcfg["path"], dcfg.get("src_key", "slop"), dcfg.get("tgt_key", "human"), dcfg.get("limit"),
                        dcfg.get("identity_frac", 0.0), seed)
    if kind == "denoise":
        return DenoiseData(dcfg["path"], dcfg["noise"], seed, dcfg.get("limit"))
    if kind == "mixed":
        return MixedData([build_dataset(p, seed) for p in dcfg["parts"]])
    raise ValueError(kind)


def param_groups(wrapper, ocfg):
    new = wrapper.new_param_names()
    groups = {"pre_decay": [], "pre_nodecay": [], "new_decay": [], "new_nodecay": []}
    seen = set()
    for n, p in wrapper.model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        nodecay = p.ndim < 2 or "norm" in n or n.endswith(".gate") or "embeddings" in n
        g = ("new_" if n in new else "pre_") + ("nodecay" if nodecay else "decay")
        groups[g].append(p)
    lr_pre, lr_new, wd = float(ocfg["lr_pre"]), float(ocfg.get("lr_new", ocfg["lr_pre"])), float(ocfg.get("weight_decay", 0.01))
    out = []
    for g, ps in groups.items():
        if ps:
            out.append({"params": ps, "lr": lr_new if g.startswith("new") else lr_pre,
                        "weight_decay": wd if g.endswith("_decay") else 0.0, "name": g})
    return out


def lr_lambda(step, total, warmup, schedule, min_ratio=0.1):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if schedule == "constant":
        return 1.0
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(wrapper, ds, max_tokens, seed, max_batches=None):
    wrapper.model.eval()
    tot, n = 0.0, 0
    batches = token_budget_batches(ds, max_tokens, epoch=0, seed=seed + 999)
    for bi, b in enumerate(batches):
        if max_batches and bi >= max_batches:
            break
        batch = wrapper.collate([ds.get(i, 0) for i in b])
        with torch.autocast(wrapper.device.type, dtype=torch.bfloat16, enabled=wrapper.device.type == "cuda"):
            loss = wrapper.loss(batch)
        k = int((batch["labels"] != -100).sum())
        tot += float(loss) * k
        n += k
    wrapper.model.train()
    return tot / max(1, n)


def save_ckpt(path, wrapper, opt, sched, state):
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    wrapper.save(tmp)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "state": state,
                "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}},
               os.path.join(tmp, "trainer_state.pt"))
    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_path(cfg, k, v)
    tcfg, ocfg = cfg["train"], cfg["optim"]
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out, "config.yaml"), "w"))
    seed = cfg.get("seed", 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and tcfg.get("device", "auto") != "cpu" else "cpu")
    use_bf16 = device.type == "cuda" and tcfg.get("bf16", True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = os.path.join(out, "ckpt")
    resuming = args.resume and os.path.exists(os.path.join(ckpt_dir, "trainer_state.pt"))
    wrapper = load_wrapper(ckpt_dir) if resuming else build_wrapper(cfg["model"])
    wrapper.to(device)
    wrapper.model.train()
    if tcfg.get("grad_ckpt", False):
        wrapper.model.gradient_checkpointing_enable()
    if tcfg.get("freeze_pretrained", False):  # adapt only new params (optional warm-up phase)
        new = wrapper.new_param_names()
        for n, p in wrapper.model.named_parameters():
            p.requires_grad = n in new

    train_ds = build_dataset(cfg["data"]["train"], seed)
    val_ds = build_dataset(cfg["data"]["val"], seed) if cfg["data"].get("val") else None
    max_tokens = int(tcfg["max_tokens"])
    accum = int(tcfg.get("grad_accum", 1))
    n_batches_epoch = len(token_budget_batches(train_ds, max_tokens, 0, seed))
    total_steps = int(tcfg.get("max_steps") or math.ceil(n_batches_epoch * float(tcfg.get("epochs", 1)) / accum))
    warmup = int(total_steps * float(ocfg.get("warmup_ratio", 0.03)))

    opt = torch.optim.AdamW(param_groups(wrapper, ocfg), betas=tuple(ocfg.get("betas", (0.9, 0.98))),
                            eps=float(ocfg.get("eps", 1e-8)), fused=device.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_steps, warmup, ocfg.get("schedule", "cosine"), float(ocfg.get("min_lr_ratio", 0.1))))
    state = {"step": 0, "epoch": 0, "batch_in_epoch": 0, "tokens": 0, "best_val": float("inf"), "bad_evals": 0,
             "wall": 0.0}
    if resuming:
        ts = torch.load(os.path.join(ckpt_dir, "trainer_state.pt"), weights_only=False)
        opt.load_state_dict(ts["opt"]); sched.load_state_dict(ts["sched"]); state = ts["state"]
        random.setstate(ts["rng"]["py"]); np.random.set_state(ts["rng"]["np"]); torch.set_rng_state(ts["rng"]["torch"])
        if ts["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ts["rng"]["cuda"])
        print(f"resumed at step {state['step']} (epoch {state['epoch']}, batch {state['batch_in_epoch']})")

    n_params = sum(p.numel() for p in {id(p): p for p in wrapper.model.parameters()}.values())
    print(f"model={wrapper.type} params={n_params / 1e6:.1f}M device={device} bf16={use_bf16} "
          f"train={len(train_ds)} batches/epoch={n_batches_epoch} accum={accum} total_steps={total_steps}", flush=True)

    mpath = os.path.join(out, "metrics.csv")
    new_file = not os.path.exists(mpath)
    mf = open(mpath, "a", newline="")
    mw = csv.writer(mf)
    if new_file:
        mw.writerow(["step", "split", "loss", "lr_pre", "lr_new", "tok_per_s", "tokens", "mem_gb", "gate_mean_abs",
                     "gate_max_abs", "wall_s", "grad_norm"])

    log_every, eval_every = int(tcfg.get("log_every", 20)), int(tcfg.get("eval_every", 500))
    save_every = int(tcfg.get("save_every", 500))
    patience = int(tcfg.get("early_stop_patience", 0))
    clip = float(ocfg.get("grad_clip", 1.0))
    t_last, tok_last, run_loss, run_n = time.time(), 0, 0.0, 0
    t_start_wall = time.time() - state["wall"]
    stop = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    while state["step"] < total_steps and not stop:
        batches = token_budget_batches(train_ds, max_tokens, state["epoch"], seed)
        while state["batch_in_epoch"] + accum <= len(batches) and state["step"] < total_steps:
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                b = batches[state["batch_in_epoch"]]
                state["batch_in_epoch"] += 1
                batch = wrapper.collate([train_ds.get(i, state["epoch"]) for i in b])
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    loss = wrapper.loss(batch) / accum
                loss.backward()
                run_loss += float(loss) * accum
                run_n += 1
                ntok = wrapper.n_tokens(batch)
                state["tokens"] += ntok
                tok_last += ntok
            gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], clip)
            opt.step()
            sched.step()
            state["step"] += 1
            s = state["step"]
            if s % log_every == 0 or s == 1:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t_last
                lrs = {g["name"]: g["lr"] for g in opt.param_groups}
                gates = wrapper.gate_values() if hasattr(wrapper, "gate_values") else {}
                gabs = [abs(v) for v in gates.values()]
                mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
                row = [s, "train", run_loss / max(1, run_n), lrs.get("pre_decay", 0), lrs.get("new_decay", ""),
                       round(tok_last / max(dt, 1e-6)), state["tokens"], round(mem, 2),
                       round(float(np.mean(gabs)), 4) if gabs else "", round(max(gabs), 4) if gabs else "",
                       round(time.time() - t_start_wall), round(float(gn), 3)]
                mw.writerow(row); mf.flush()
                eta = (total_steps - s) * dt / max(1, (log_every if s > 1 else 1)) / 60
                print(f"step {s}/{total_steps} loss {row[2]:.4f} tok/s {row[5]} mem {row[7]}GB "
                      f"gates {row[8]} gn {row[11]} | ETA {eta:.1f} min", flush=True)
                t_last, tok_last, run_loss, run_n = time.time(), 0, 0.0, 0
            if val_ds is not None and (s % eval_every == 0 or s == total_steps):
                vl = evaluate(wrapper, val_ds, max_tokens, seed, tcfg.get("eval_batches"))
                mw.writerow([s, "val", vl, "", "", "", state["tokens"], "", "", "", round(time.time() - t_start_wall), ""])
                mf.flush()
                improved = vl < state["best_val"] - 1e-4
                print(f"  val loss {vl:.4f} (best {min(vl, state['best_val']):.4f}){' *' if improved else ''}", flush=True)
                if improved:
                    state["best_val"], state["bad_evals"] = vl, 0
                    wrapper.save(os.path.join(out, "best"))
                else:
                    state["bad_evals"] += 1
                    if patience and state["bad_evals"] >= patience:
                        print("early stopping", flush=True)
                        stop = True
            if s % save_every == 0 or stop or s == total_steps:
                state["wall"] = time.time() - t_start_wall
                save_ckpt(ckpt_dir, wrapper, opt, sched, state)
            if stop:
                break
        if not stop and state["step"] < total_steps:
            state["epoch"] += 1
            state["batch_in_epoch"] = 0

    state["wall"] = time.time() - t_start_wall
    save_ckpt(ckpt_dir, wrapper, opt, sched, state)
    if val_ds is None:
        wrapper.save(os.path.join(out, "best"))
    summary = {"steps": state["step"], "tokens": state["tokens"], "best_val": state["best_val"],
               "wall_min": round(state["wall"] / 60, 1), "params_M": round(n_params / 1e6, 1),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
               "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("done", summary)


if __name__ == "__main__":
    main()
