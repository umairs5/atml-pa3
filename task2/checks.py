"""Interface checks on the real data: feature shapes, frozen/trainable sets, connector gradients,
split isolation and the fixed mismatched-image mapping.

python -m task2.checks --features FEATS --splits SPLITS --out RESULTS/checks.json
"""
import argparse
import os

import numpy as np
import torch

from common.utils import load_json, save_json
from task2.train import caption_example, pair_examples, set_trainable
from task2.vlm import (Features, Prompter, add_lora_adapters, build_batch, build_vlm, load_tokenizer,
                       per_example_mean, token_nll)

CFG = {"seed": 6304, "visual_tokens": 196,
       "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]}}


def split_checks(splits):
    j = lambda n: load_json(os.path.join(splits, f"{n}.json"))
    ims = lambda pairs: {it["image_id"] for p in pairs for it in p["items"]}
    train = ({c["image_id"] for c in j("captions_train")} | ims(j("grounding_train_presence"))
             | ims(j("grounding_train_counting")) | ims(j("vqa_train")))
    evals = {"captions_dev": {c["image_id"] for c in j("captions_dev")}}
    for n in ["grounding_dev", "grounding_test", "vqa_dev", "vqa_test"]:
        evals[n] = ims(j(n))
    out = {"train_images": len(train), "eval_images": {k: len(v) for k, v in evals.items()},
           "train_eval_overlap": {k: len(train & v) for k, v in evals.items()}}
    names = list(evals)
    out["eval_eval_overlap"] = {f"{a}|{b}": len(evals[a] & evals[b])
                                for i, a in enumerate(names) for b in names[i + 1:]}
    bad = 0
    for n in ["grounding_dev", "grounding_test", "vqa_dev", "vqa_test"]:
        for p in j(n):
            own = {it["image_id"] for it in p["items"]}
            bad += sum(it["mismatch_image_id"] in own or it["mismatch_image_id"] not in evals[n] for it in p["items"])
    out["mismatch_violations"] = bad
    counts = {n: len(j(n)) for n in ["captions_train", "captions_dev", "grounding_train_presence",
                                    "grounding_train_counting", "grounding_dev", "grounding_test",
                                    "vqa_train", "vqa_dev", "vqa_test"]}
    out["sizes"] = counts
    out["ok"] = (all(v == 0 for v in out["train_eval_overlap"].values())
                 and all(v == 0 for v in out["eval_eval_overlap"].values()) and bad == 0)
    return out


def param_sets(model):
    tr = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fr = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    return {"trainable": sum(p.numel() for p in tr.values()), "frozen": sum(p.numel() for p in fr.values()),
            "trainable_groups": sorted({n.split(".")[0] if n.startswith("connector") else n.split(".")[-1] for n in tr}),
            "trainable_dtypes": sorted({str(p.dtype) for p in tr.values()}),
            "frozen_dtypes": sorted({str(p.dtype) for p in fr.values()}),
            "only_connector_or_lora": all(n.startswith("connector.") or "lora_" in n for n in tr)}


def grad_check(model, prompter, feats, data, device):
    exs = [caption_example(c) for c in data["captions_train"][:2]]
    exs += pair_examples(data["vqa_train"][0], "vqa")
    batch = build_batch(prompter, exs, feats, 196, device)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        nll, ex, _ = token_nll(model, batch)
        loss = per_example_mean(nll.float(), ex, len(exs)).mean()
    loss.backward()
    g = {n: float(p.grad.norm()) for n, p in model.connector.named_parameters()}
    leaked = [n for n, p in model.lm.named_parameters() if not p.requires_grad and p.grad is not None]
    lora_g = [float(p.grad.norm()) for n, p in model.lm.named_parameters() if "lora_" in n and p.grad is not None]
    model.zero_grad(set_to_none=True)
    return {"loss": float(loss), "connector_grad_norms": g, "frozen_params_with_grad": len(leaked),
            "lora_params_with_grad": len(lora_g), "ok": all(v > 0 for v in g.values()) and not leaked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--splits", default="splits/task2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = {"splits": split_checks(args.splits)}
    info = load_json(os.path.join(args.features, "feature_info.json"))
    arr = np.load(os.path.join(args.features, "features.npy"), mmap_mode="r")
    out["features"] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                       "post_layernorm_applied_once": info["post_layernorm_applied_once"],
                       "finite": bool(np.isfinite(arr[:256].astype(np.float32)).all()),
                       "mean_token_norm_first_256": float(np.linalg.norm(arr[:256].astype(np.float32), axis=-1).mean())}

    data = {n: load_json(os.path.join(args.splits, f"{n}.json")) for n in ["captions_train", "vqa_train"]}
    feats = Features(args.features)
    prompter = Prompter(load_tokenizer())
    model = build_vlm(CFG, device, with_lora=False, seed=6304)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    set_trainable(model, 0)
    out["stage1_params"] = param_sets(model)
    out["stage1_grad"] = grad_check(model, prompter, feats, data, device)
    add_lora_adapters(model, CFG)
    set_trainable(model, 1)
    out["stage2_params"] = param_sets(model)
    out["stage2_grad"] = grad_check(model, prompter, feats, data, device)
    save_json(out, args.out)
    for k, v in out.items():
        print(k, v)
    assert out["splits"]["ok"] and out["stage1_grad"]["ok"] and out["stage2_grad"]["ok"]


if __name__ == "__main__":
    main()
