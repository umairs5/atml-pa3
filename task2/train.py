"""Task 2 training: caption alignment -> visual instruction tuning -> natural VQA, one continuous run.

python -m task2.train --config task2/configs/vlm196.yaml --features FEATS --out_root RUNS

Re-running the same command resumes from the latest checkpoint (every 250 updates).
"""
import argparse
import glob
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from common.lora import lora_layers
from common.utils import (count_params, get_rng_state, load_json, load_yaml, save_json,
                          set_rng_state, set_seed)
from task2.vlm import (Features, Prompter, add_lora_adapters, build_batch, build_vlm,
                       load_tokenizer, load_trainable, per_example_mean, token_nll, trainable_state)

SPLITS = "splits/task2"


def load_data(splits):
    d = {k: load_json(os.path.join(splits, f"{k}.json")) for k in
         ["captions_train", "captions_dev", "grounding_train_presence", "grounding_train_counting",
          "grounding_dev", "vqa_train", "vqa_dev"]}
    return d


def caption_example(c):
    return {"kind": "caption", "image_id": c["image_id"], "target": c["caption"]}


def pair_examples(p, kind):
    """The two question-image examples of a pair (presence, counting or vqa)."""
    out = []
    for it in p["items"]:
        tgt = it["target"] if kind == "vqa" else it["answer"]
        out.append({"kind": kind, "question": p["question"], "image_id": it["image_id"], "target": tgt})
    return out


def stage_bounds(cfg):
    bounds, s = [], 0
    for st in cfg["stages"]:
        bounds.append((s, s + st["updates"]))
        s += st["updates"]
    return bounds


def stage_of(cfg, u):
    """Stage index of 0-based update u."""
    for k, (a, b) in enumerate(stage_bounds(cfg)):
        if a <= u < b:
            return k
    raise ValueError(u)


def draws_for_update(cfg, data, u):
    """Examples of update u (0-based). Stateless: a function of the seed and u only, so the
    sampler position after a resume is just the update index."""
    k = stage_of(cfg, u)
    st = cfg["stages"][k]
    eff = cfg["micro_batch"] * cfg["grad_accum"]
    caps = data["captions_train"]
    if st["mixture"] == "passes":
        # stage 1: shuffled passes over all 18,000 caption examples
        a, _ = stage_bounds(cfg)[k]
        start = (u - a) * eff
        p = start // len(caps)
        perm = np.random.default_rng([cfg["seed"], 10, p]).permutation(len(caps))
        idx = perm[start % len(caps):start % len(caps) + eff]
        assert len(idx) == eff
        return [caption_example(caps[i]) for i in idx], {"caption": eff}
    rng = np.random.default_rng([cfg["seed"], 20, u])
    comps = list(st["mixture"])
    probs = np.array([st["mixture"][c] for c in comps], dtype=float)
    pools = {"presence": data["grounding_train_presence"], "counting": data["grounding_train_counting"],
             "vqa": data["vqa_train"]}
    exs, mix = [], {}  # mix counts examples, two per draw
    for _ in range(eff // 2):  # each draw is a pair, or two caption examples
        c = comps[rng.choice(len(comps), p=probs)]
        mix[c] = mix.get(c, 0) + 2
        if c == "caption":
            exs += [caption_example(caps[i]) for i in rng.integers(len(caps), size=2)]
        else:
            exs += pair_examples(pools[c][rng.integers(len(pools[c]))], c)
    return exs, mix


def lr_lambda(n, warmup, min_frac):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, n - warmup - 1)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def make_optimizer(cfg, model, k):
    st = cfg["stages"][k]
    conn = list(model.connector.parameters())
    groups = [{"params": [p for p in conn if p.ndim >= 2], "lr": st["connector_lr"], "weight_decay": cfg["weight_decay"]},
              {"params": [p for p in conn if p.ndim < 2], "lr": st["connector_lr"], "weight_decay": 0.0}]
    if st.get("lora_lr"):
        lora = [p for _, m in lora_layers(model.lm) for p in (m.lora_A, m.lora_B)]
        groups.append({"params": lora, "lr": st["lora_lr"], "weight_decay": cfg["weight_decay"]})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    warm = math.ceil(cfg["warmup_frac"] * st["updates"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(st["updates"], warm, cfg["min_lr_frac"]))
    return opt, sched, warm


def set_trainable(model, k):
    model.lm.requires_grad_(False)
    model.connector.requires_grad_(True)
    for _, m in lora_layers(model.lm):
        m.lora_A.requires_grad_(True)
        m.lora_B.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def calibrate_scale(model, data, feats, prompter, device, n_images=512, seed=6304):
    """Fixed connector output scale from training data only: mean L2 norm of the text-token
    embeddings in the training captions, questions and answers, divided by the mean L2 norm
    of the connector's initial outputs on training images."""
    emb = model.lm.get_input_embeddings().weight
    counts = torch.zeros(emb.shape[0], dtype=torch.float64)
    texts = [c["caption"] for c in data["captions_train"]]
    for kind in ["grounding_train_presence", "grounding_train_counting", "vqa_train"]:
        for p in data[kind]:
            texts.append(p["question"])
            texts += [it["target"] if kind == "vqa_train" else it["answer"] for it in p["items"]]
    for t in texts:
        ids = prompter.tok(t, add_special_tokens=False).input_ids
        counts += torch.bincount(torch.tensor(ids), minlength=emb.shape[0]).double()
    norms = emb.float().norm(dim=-1).double().cpu()
    text_norm = float((counts * norms).sum() / counts.sum())

    rng = np.random.default_rng([seed, 30])
    ids = sorted({c["image_id"] for c in data["captions_train"]})
    pick = [ids[i] for i in rng.choice(len(ids), size=min(n_images, len(ids)), replace=False)]
    model.connector.scale.fill_(1.0)
    vis = []
    for s in range(0, len(pick), 64):
        f = feats.get(pick[s:s + 64], device).float()
        vis.append(model.connector(f).norm(dim=-1).flatten().cpu())
    vis_norm = float(torch.cat(vis).double().mean())
    scale = text_norm / vis_norm
    model.connector.scale.fill_(scale)
    return {"rule": "connector output x scale, scale = mean text-embedding norm (over all tokens of the training "
                    "captions, questions and answers) / mean norm of initial connector outputs on "
                    f"{len(pick)} training images",
            "text_norm": text_norm, "visual_norm_at_init": vis_norm, "scale": scale,
            "n_text_tokens": int(counts.sum())}


@torch.no_grad()
def dev_eval(model, data, feats, prompter, cfg, device, use_amp, bs=16):
    """Caption reference loss and reference-answer NLL under correct / complementary / no image."""
    model.eval()
    nv = cfg["visual_tokens"]

    def nll_of(examples, drop_end):
        vals = []
        for s in range(0, len(examples), bs):
            chunk = examples[s:s + bs]
            batch = build_batch(prompter, chunk, feats, nv, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                nll, ex, end = token_nll(model, batch)
            m = per_example_mean(nll.float(), ex, len(chunk), select=~end if drop_end else None)
            vals += m.tolist()
        return vals

    out = {"caption_loss": float(np.mean(nll_of([caption_example(c) for c in data["captions_dev"]], False)))}
    for suite, kindmap in [("grounding", None), ("vqa", "vqa")]:
        pairs = data[f"{suite}_dev"]
        res = {}
        for cond in ["correct", "complementary", "none"]:
            exs = []
            for p in pairs:
                kind = kindmap or p["kind"]
                base = pair_examples(p, kind)
                for s, ex in enumerate(base):
                    img = {"correct": p["items"][s]["image_id"], "complementary": p["items"][1 - s]["image_id"],
                           "none": None}[cond]
                    exs.append({**ex, "image_id": img})
            res[cond] = nll_of(exs, True)
        out[suite] = {f"nll_{c}": float(np.mean(v)) for c, v in res.items()}
        out[suite]["swap_penalty"] = float(np.mean(np.array(res["complementary"]) - np.array(res["correct"])))
    model.train()
    return out


def latest_checkpoint(out):
    found = sorted(glob.glob(os.path.join(out, "ckpt-*", "recovery.pt")))
    return found[-1] if found else None


def truncate_log(path, upto):
    if os.path.exists(path):
        with open(path) as f:
            lines = [l for l in f if json.loads(l)["update"] <= upto]
        with open(path, "w") as f:
            f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", required=True, help="folder with features.npy and index.json")
    ap.add_argument("--out_root", default="outputs/task2/runs")
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--max_updates", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--start_stage", type=int, default=None, help="smoke tests: jump to this stage")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = os.path.join(args.out_root, cfg["name"])
    os.makedirs(out, exist_ok=True)
    save_json(cfg, os.path.join(out, "config.json"))
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    data = load_data(args.splits)
    feats = Features(args.features)
    tok = load_tokenizer()
    prompter = Prompter(tok)
    bounds = stage_bounds(cfg)
    total = bounds[-1][1]
    eval_at = set(range(0, total + 1, cfg["eval_every"])) | {b for _, b in bounds}
    eff = cfg["micro_batch"] * cfg["grad_accum"]

    ck = latest_checkpoint(out)
    state = torch.load(ck, map_location="cpu", weights_only=False) if ck else None
    start = state["update"] if state else 0
    if args.start_stage is not None and state is None:
        start = bounds[args.start_stage][0]
    first_stage = stage_of(cfg, min(start, total - 1))
    model = build_vlm(cfg, device, with_lora=first_stage >= 1, seed=cfg["seed"])
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    log_path, dev_path = os.path.join(out, "train_log.jsonl"), os.path.join(out, "dev_log.jsonl")
    if state:
        load_trainable(model, state["trainable"])
        res = state["resources"]
        truncate_log(log_path, start)
        truncate_log(dev_path, start)
        print(f"resumed from {ck} at update {start}")
    else:
        for p in (log_path, dev_path):
            if os.path.exists(p):
                os.remove(p)
        calib = calibrate_scale(model, data, feats, prompter, device, seed=cfg["seed"])
        save_json(calib, os.path.join(out, "calibration.json"))
        print("calibration:", calib)
        res = {"name": cfg["name"], "train_seconds": 0.0, "dev_eval_seconds": 0.0,
               "peak_mem_allocated_gb": 0.0, "peak_mem_reserved_gb": 0.0, "sessions": [],
               "skipped_updates": [], "trainable_params": {}, "mixture_counts": {}, "calibration": calib}

    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    if state:
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])

    end = total if args.max_updates is None else min(total, start + args.max_updates)
    session = {"gpu": torch.cuda.get_device_name() if use_amp else "cpu", "start_update": start, "end_update": start}
    res["sessions"].append(session)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    if start == 0 and 0 in eval_at and not state:
        t0 = time.time()
        d = dev_eval(model, data, feats, prompter, cfg, device, use_amp)
        res["dev_eval_seconds"] += time.time() - t0
        with open(dev_path, "a") as f:
            f.write(json.dumps({"update": 0, "stage": 0, **d}) + "\n")
        print("dev @0:", d)

    opt = sched = None
    cur_stage = None
    log_f = open(log_path, "a")
    for u in range(start, end):
        k = stage_of(cfg, u)
        if k != cur_stage:
            if k >= 1 and not lora_layers(model.lm):
                add_lora_adapters(model, cfg)
            trainable = set_trainable(model, k)
            opt, sched, warm = make_optimizer(cfg, model, k)
            if state and state.get("stage") == k and state["update"] == u:
                # resumed inside this stage: restore its optimizer and schedule
                opt.load_state_dict(state["optimizer"])
                sched.load_state_dict(state["scheduler"])
            res["trainable_params"][cfg["stages"][k]["name"]] = count_params(trainable)
            res.setdefault("warmup_updates", {})[cfg["stages"][k]["name"]] = warm
            print(f"stage {k + 1} ({cfg['stages'][k]['name']}): {count_params(trainable):,} trainable params")
            cur_stage = k
            state = None

        t0 = time.time()
        exs, mix = draws_for_update(cfg, data, u)
        for c, n in mix.items():
            res["mixture_counts"].setdefault(cfg["stages"][k]["name"], {})
            res["mixture_counts"][cfg["stages"][k]["name"]][c] = res["mixture_counts"][cfg["stages"][k]["name"]].get(c, 0) + n
        lrs = [g["lr"] for g in opt.param_groups]
        loss_val, n_tok = 0.0, 0
        for j in range(0, eff, cfg["micro_batch"]):
            batch = build_batch(prompter, exs[j:j + cfg["micro_batch"]], feats, cfg["visual_tokens"], device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                nll, ex, _ = token_nll(model, batch)
                # mean over each example's supervised tokens, then over the effective batch
                loss = per_example_mean(nll.float(), ex, len(exs[j:j + cfg["micro_batch"]])).sum() / eff
            scaler.scale(loss).backward()
            loss_val += loss.item()
            n_tok += nll.numel()

        scaler.unscale_(opt)
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
        rec = {"update": u + 1, "stage": k + 1, "loss": loss_val, "lr_connector": lrs[0],
               "lr_lora": lrs[2] if len(lrs) > 2 else None, "grad_norm": float(gnorm), "scale": scale,
               "skipped": skipped, "mix": mix, "sup_tokens": n_tok, "sec": round(dt, 3)}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if (u + 1) % 25 == 0 or u == start:
            print(f"[{cfg['name']}] update {u + 1}/{total} stage {k + 1} loss {loss_val:.4f} "
                  f"lr {lrs[0]:.2e} gnorm {float(gnorm):.3f} {dt:.2f}s", flush=True)
        if use_amp:
            res["peak_mem_allocated_gb"] = max(res["peak_mem_allocated_gb"], torch.cuda.max_memory_allocated() / 2**30)
            res["peak_mem_reserved_gb"] = max(res["peak_mem_reserved_gb"], torch.cuda.max_memory_reserved() / 2**30)
        session["end_update"] = u + 1

        if (u + 1) in eval_at:
            t1 = time.time()
            d = dev_eval(model, data, feats, prompter, cfg, device, use_amp)
            res["dev_eval_seconds"] += time.time() - t1
            with open(dev_path, "a") as f:
                f.write(json.dumps({"update": u + 1, "stage": k + 1, **d}) + "\n")
            print(f"dev @{u + 1}:", json.dumps(d))

        if (u + 1) == bounds[k][1]:
            # connector / adapter weights at the end of each stage
            sd = os.path.join(out, f"stage{k + 1}")
            os.makedirs(sd, exist_ok=True)
            torch.save(trainable_state(model), os.path.join(sd, "weights.pt"))
            print(f"saved stage {k + 1} weights")

        if (u + 1) % cfg["save_every"] == 0 or (u + 1) == end:
            save_checkpoint(out, u + 1, k, model, opt, sched, scaler, res)

    log_f.close()
    if end == total:
        res["done"] = True
    save_json(res, os.path.join(out, "resources.json"))
    print(json.dumps({k: v for k, v in res.items() if k != "calibration"}, indent=1))


def save_checkpoint(out, update, stage, model, opt, sched, scaler, res):
    d = os.path.join(out, f"ckpt-{update:04d}")
    os.makedirs(d, exist_ok=True)
    state = {"update": update, "stage": stage, "trainable": trainable_state(model),
             "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
             "rng": get_rng_state(), "resources": res}
    tmp = os.path.join(d, "recovery.pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, os.path.join(d, "recovery.pt"))
    save_json(res, os.path.join(out, "resources.json"))
    # keep only the two newest recovery checkpoints, stage weights are saved separately
    for old in sorted(glob.glob(os.path.join(out, "ckpt-*", "recovery.pt")))[:-2]:
        os.remove(old)
        os.rmdir(os.path.dirname(old))


if __name__ == "__main__":
    main()
