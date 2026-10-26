"""Task 1 training: full FT, full FT + L2-SP, LoRA and OPLoRA on MetaMathQA.

python -m task1.train --config task1/configs/full_ft.yaml --out_root /content/drive/MyDrive/pa3/task1

Re-running the same command resumes from the latest checkpoint in the run folder.
"""
import argparse
import glob
import json
import os
import time

# full FT + L2-SP runs close to the 15 GB of a T4, fragmentation alone can cause an OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

from common.lora import add_lora, export_factors
from common.utils import (count_params, get_rng_state, load_yaml, path_size, save_json,
                          set_rng_state, set_seed)

MODEL = "Qwen/Qwen3-0.6B"


class MetaMathData:
    def __init__(self, path, pad_id):
        d = np.load(path)
        self.tokens = d["tokens"]
        self.offsets = d["offsets"]
        self.plens = d["prompt_lens"]
        self.pad_id = pad_id

    def __len__(self):
        return len(self.plens)

    def n_response_tokens(self, idx):
        return int(sum(self.offsets[i + 1] - self.offsets[i] - self.plens[i] for i in idx))

    def batch(self, idx, device):
        seqs = [self.tokens[self.offsets[i]:self.offsets[i + 1]] for i in idx]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for j, (i, s) in enumerate(zip(idx, seqs)):
            n = len(s)
            ids[j, :n] = torch.from_numpy(s.astype(np.int64))
            mask[j, :n] = 1
            labels[j, self.plens[i]:n] = ids[j, self.plens[i]:n]
        return ids.to(device), mask.to(device), labels.to(device)


def _ce_sum(h, w, y):
    return F.cross_entropy(F.linear(h, w).float(), y, reduction="sum")


def response_loss_sum(model, ids, mask, labels, chunk=1024):
    """Summed next-token CE over supervised positions only.

    The LM head is applied only where labels exist, in checkpointed chunks, so the
    full (batch, seq, 151k) logit tensor is never materialized.
    """
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    tgt = labels[:, 1:]
    keep = tgt != -100
    h, y = hidden[:, :-1][keep], tgt[keep]
    w = model.lm_head.weight
    total = torch.zeros((), device=ids.device)
    for s in range(0, y.numel(), chunk):
        total = total + checkpoint(_ce_sum, h[s:s + chunk], w, y[s:s + chunk], use_reentrant=False)
    return total


@torch.no_grad()
def l2sp_grad(params, ref, lam):
    """Add lam * (theta - theta0) to the (already unscaled) grads, return Omega."""
    sq = torch.zeros((), dtype=torch.float64, device=params[0].device)
    for p, p0 in zip(params, ref):
        d = p - p0.to(p.device, non_blocking=True)
        sq += torch.linalg.vector_norm(d).double() ** 2
        if p.grad is not None:
            p.grad.add_(d, alpha=lam)
    return 0.5 * sq.item()


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def latest_checkpoint(out):
    found = sorted(glob.glob(os.path.join(out, "ckpt-*", "recovery.pt")))
    return found[-1] if found else None


def save_checkpoint(out, update, model, opt, sched, scaler, res, cfg, full):
    d = os.path.join(out, f"ckpt-{update:04d}")
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    state = {
        "update": update,
        "examples_seen": update * cfg["micro_batch"] * cfg["grad_accum"],
        "trainable": trainable_state(model),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": get_rng_state(),
        "resources": res,
    }
    tmp = os.path.join(d, "recovery.pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, os.path.join(d, "recovery.pt"))
    info = {"recovery_bytes": path_size(os.path.join(d, "recovery.pt")),
            "save_seconds": round(time.time() - t0, 1)}

    if update in cfg["export_updates"]:
        t0 = time.time()
        if full:
            # fp32 so later weight diffs are not swamped by fp16 rounding
            model.config.use_cache = True
            model.save_pretrained(os.path.join(d, "model"), safe_serialization=True)
            model.config.use_cache = False
            info["export_bytes"] = path_size(os.path.join(d, "model"))
        else:
            # projected factors for OPLoRA, plain factors for LoRA
            torch.save(export_factors(model), os.path.join(d, "adapter.pt"))
            info["export_bytes"] = path_size(os.path.join(d, "adapter.pt"))
        info["export_seconds"] = round(time.time() - t0, 1)

    res["checkpoints"][str(update)] = info
    save_json(res, os.path.join(out, "resources.json"))
    print(f"saved checkpoint {update}: {info}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_root", default="outputs/task1/runs")
    ap.add_argument("--data", default="outputs/task1/data/metamath_40k.npz")
    ap.add_argument("--max_updates", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--n_examples", type=int, default=None, help="cap examples per sequence for CPU tests")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = os.path.join(args.out_root, cfg["name"])
    os.makedirs(out, exist_ok=True)
    save_json(cfg, os.path.join(out, "config.json"))

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    full = cfg["method"] == "full"

    # full FT keeps fp32 master weights; adapter runs use a frozen fp16 backbone
    dtype = torch.float32 if (full or not use_amp) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device)
    model.config.use_cache = False

    res = {"method": cfg["method"], "name": cfg["name"], "train_seconds": 0.0,
           "peak_mem_allocated_gb": 0.0, "peak_mem_reserved_gb": 0.0, "sessions": [],
           "skipped_updates": [], "checkpoints": {}}

    if not full:
        model.requires_grad_(False)
        lc = cfg["lora"]
        wrapped, svd_sec = add_lora(model, lc["targets"], lc["r"], lc["alpha"], lc["dropout"],
                                    seed=cfg["seed"], protect_rank=lc.get("protect_rank", 0))
        res["n_adapted_modules"] = len(wrapped)
        if lc.get("protect_rank", 0):
            res["svd_seconds"] = round(svd_sec, 2)
            res["basis_bytes"] = sum((m.U_k.numel() + m.V_k.numel()) * 4 for _, m in wrapped)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    res["trainable_params"] = count_params(trainable)
    res["total_params"] = count_params(model.parameters())

    # theta_0 reference has to be taken before any checkpoint is loaded
    lam = cfg.get("l2sp_lambda", 0.0)
    ref = None
    if lam > 0:
        ref_dev = cfg.get("ref_device", device)
        if ref_dev == "cpu":
            # pinned host copy, streamed to the GPU one tensor at a time in l2sp_grad
            ref = [p.detach().cpu().clone().pin_memory() if use_amp else p.detach().cpu().clone()
                   for p in trainable]
        else:
            ref = [p.detach().clone().to(ref_dev) for p in trainable]
        res["ref_bytes"] = sum(r.numel() * r.element_size() for r in ref)
        res["ref_device"] = ref_dev
        res["l2sp_seconds"] = 0.0

    decay = [p for p in trainable if p.ndim >= 2]
    no_decay = [p for p in trainable if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8)
    sched = get_cosine_schedule_with_warmup(opt, cfg["warmup"], cfg["updates"])
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    start = 0
    ck = latest_checkpoint(out)
    log_path = os.path.join(out, "train_log.jsonl")
    if ck:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        params = dict(model.named_parameters())
        with torch.no_grad():
            for n, t in state["trainable"].items():
                params[n].copy_(t)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        start = state["update"]
        res = state["resources"]
        # drop log lines written after that checkpoint
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = [l for l in f if json.loads(l)["update"] <= start]
            with open(log_path, "w") as f:
                f.writelines(lines)
        print(f"resumed from {ck} at update {start}")
        del state
    elif os.path.exists(log_path):
        os.remove(log_path)  # leftovers from a run that died before its first checkpoint

    tok_pad = 151643  # <|endoftext|>, Qwen's pad token
    data = MetaMathData(args.data, tok_pad)
    eff = cfg["micro_batch"] * cfg["grad_accum"]
    assert len(data) >= cfg["updates"] * eff or args.max_updates

    end = cfg["updates"] if args.max_updates is None else min(cfg["updates"], args.max_updates)
    session = {"gpu": torch.cuda.get_device_name() if use_amp else "cpu",
               "start_update": start, "end_update": start}
    res["sessions"].append(session)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    log_f = open(log_path, "a")
    for u in range(start, end):
        t0 = time.time()
        idx = list(range(u * eff, (u + 1) * eff))
        if args.n_examples:
            idx = idx[:args.n_examples]
        n_tok = data.n_response_tokens(idx)
        lr = sched.get_last_lr()[0]

        sft = 0.0
        for j in range(0, len(idx), cfg["micro_batch"]):
            ids, mask, labels = data.batch(idx[j:j + cfg["micro_batch"]], device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = response_loss_sum(model, ids, mask, labels) / n_tok
            scaler.scale(loss).backward()
            sft += loss.item()

        scaler.unscale_(opt)
        omega = None
        if ref is not None:
            t1 = time.time()
            omega = l2sp_grad(trainable, ref, lam)
            res["l2sp_seconds"] += time.time() - t1
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg["clip"])
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = bool(use_amp and scaler.get_scale() < scale)
        sched.step()
        opt.zero_grad(set_to_none=True)

        dt = time.time() - t0
        res["train_seconds"] += dt
        if skipped:
            res["skipped_updates"].append(u + 1)
        rec = {"update": u + 1, "examples": (u + 1) * eff, "sft_loss": sft, "lr": lr,
               "grad_norm": float(gnorm), "scale": scale, "skipped": skipped,
               "resp_tokens": n_tok, "sec": round(dt, 3)}
        if omega is not None:
            rec.update(omega=omega, lam_omega=lam * omega, total_loss=sft + lam * omega)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if (u + 1) % 10 == 0 or u == start:
            print(f"[{cfg['name']}] update {u + 1}/{cfg['updates']} loss {sft:.4f} lr {lr:.2e} "
                  f"gnorm {float(gnorm):.3f} scale {scale:.0f} {dt:.1f}s"
                  + (f" omega {omega:.4f}" if omega is not None else ""))

        if use_amp:
            res["peak_mem_allocated_gb"] = max(res["peak_mem_allocated_gb"],
                                               torch.cuda.max_memory_allocated() / 2**30)
            res["peak_mem_reserved_gb"] = max(res["peak_mem_reserved_gb"],
                                              torch.cuda.max_memory_reserved() / 2**30)
        session["end_update"] = u + 1
        if (u + 1) in cfg["save_updates"]:
            save_checkpoint(out, u + 1, model, opt, sched, scaler, res, cfg, full)

    log_f.close()
    if end == cfg["updates"]:
        res["done"] = True
    save_json(res, os.path.join(out, "resources.json"))
    print(json.dumps({k: v for k, v in res.items() if k != "checkpoints"}, indent=1))


if __name__ == "__main__":
    main()
