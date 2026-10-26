"""Weight-space and activation diagnostics (down_proj, layers 0/13/27), plus
full-model displacement for the full FT checkpoints. Everything in fp32.

python -m task1.diagnostics --runs_dir RUNS --out RESULTS/task1
"""
import argparse
import glob
import os

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import load_json, save_json

MODEL = "Qwen/Qwen3-0.6B"
LAYERS = [0, 13, 27]
K = 128
CHECKPOINTS = {
    "full_mid": ("full_ft", 625),
    "full_final": ("full_ft", 1250),
    "l2sp_final": ("full_ft_l2sp", 1250),
    "lora4_final": ("lora_r4", 1250),
    "lora16_final": ("lora_r16", 1250),
    "oplora16_final": ("oplora_r16", 1250),
}


def down_key(layer):
    return f"model.layers.{layer}.mlp.down_proj"


class SafeDir:
    """Read single tensors from a (possibly sharded) safetensors folder or file."""

    def __init__(self, path):
        files = [path] if path.endswith(".safetensors") else sorted(glob.glob(os.path.join(path, "*.safetensors")))
        self.handles = [safe_open(f, framework="pt") for f in files]
        self.where = {k: h for h in self.handles for k in h.keys()}

    def keys(self):
        return set(self.where)

    def get(self, key):
        return self.where[key].get_tensor(key).float()


def base_weights():
    return SafeDir(hf_hub_download(MODEL, "model.safetensors"))


def load_delta(runs_dir, run, update, layer, base):
    ck = os.path.join(runs_dir, run, f"ckpt-{update:04d}")
    key = down_key(layer)
    if os.path.isdir(os.path.join(ck, "model")):
        return SafeDir(os.path.join(ck, "model")).get(key + ".weight") - base.get(key + ".weight")
    f = torch.load(os.path.join(ck, "adapter.pt"), map_location="cpu")[key]
    return f["scaling"] * (f["B"] @ f["A"])


@torch.no_grad()
def displacement(runs_dir, run, update, base):
    trained = SafeDir(os.path.join(runs_dir, run, f"ckpt-{update:04d}", "model"))
    shared = trained.keys() & base.keys()
    sq = sum(torch.sum((trained.get(k).double() - base.get(k).double()) ** 2).item() for k in shared)
    return {"l2": sq ** 0.5, "n_tensors": len(shared),
            "only_in_trained": sorted(trained.keys() - base.keys()),
            "only_in_base": sorted(base.keys() - trained.keys())}


@torch.no_grad()
def cache_inputs(probes, device):
    """down_proj inputs of the starting model at the last prompt token."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device).eval()
    cache = {l: [] for l in LAYERS}
    hooks = []
    for l in LAYERS:
        mod = model.model.layers[l].mlp.down_proj
        hooks.append(mod.register_forward_hook(
            lambda m, inp, out, l=l: cache[l].append(inp[0][0, -1].float().cpu())))
    for p in probes:
        # one prompt at a time, so the last position is the last real prompt token
        ids = tok(p["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
        model.model(**ids)
    for h in hooks:
        h.remove()
    return {l: torch.stack(v) for l, v in cache.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--probes", default="splits/task1_probe_set.json")
    ap.add_argument("--out", default="outputs/task1/results")
    ap.add_argument("--checkpoints", default=None,
                    help="override, e.g. 'test=smoke_oplora:3' (comma separated)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.checkpoints:
        CHECKPOINTS.clear()
        for item in args.checkpoints.split(","):
            cond, rest = item.split("=")
            run, update = rest.split(":")
            CHECKPOINTS[cond] = (run, int(update))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    probes = load_json(args.probes)["probes"]
    sets = [p["set"] for p in probes]
    Z = cache_inputs(probes, device)
    base = base_weights()

    bases = {}
    for l in LAYERS:
        W0 = base.get(down_key(l) + ".weight").to(device)
        U, _, Vh = torch.linalg.svd(W0, full_matrices=False)
        bases[l] = (W0, U[:, :K], Vh[:K].T)

    weight_rows, act_rows = [], []
    for cond, (run, update) in CHECKPOINTS.items():
        if not os.path.isdir(os.path.join(args.runs_dir, run, f"ckpt-{update:04d}")):
            print(f"skipping {cond}: checkpoint not found")
            continue
        for l in LAYERS:
            W0, Uk, Vk = bases[l]
            dW = load_delta(args.runs_dir, run, update, l, base).to(device)
            fro2 = dW.pow(2).sum()
            weight_rows.append({
                "condition": cond, "layer": l,
                "rho_k": ((Uk.T @ dW).pow(2).sum() / fro2).item(),
                # input-side counterpart, mainly a numerical check for OPLoRA
                "rho_k_input": ((dW @ Vk).pow(2).sum() / fro2).item(),
                "eta": (fro2.sqrt() / W0.norm()).item(),
                "dW_fro": fro2.sqrt().item(),
            })
            z = Z[l].to(device)
            R = (z @ dW.T).pow(2).sum(1) / ((z @ W0.T).pow(2).sum(1) + 1e-8)
            for i, r in enumerate(R.tolist()):
                act_rows.append({"condition": cond, "layer": l, "probe": i, "set": sets[i], "R": r})

    disp = {}
    for cond in [c for c in ["full_mid", "full_final", "l2sp_final"] if c in CHECKPOINTS]:
        run, update = CHECKPOINTS[cond]
        if os.path.isdir(os.path.join(args.runs_dir, run, f"ckpt-{update:04d}", "model")):
            disp[cond] = displacement(args.runs_dir, run, update, base)

    wdf = pd.DataFrame(weight_rows)
    adf = pd.DataFrame(act_rows)
    wdf.to_csv(os.path.join(args.out, "weight_diagnostics.csv"), index=False)
    adf.to_csv(os.path.join(args.out, "activation_ratios.csv"), index=False)
    summary = (adf.groupby(["condition", "layer", "set"])["R"]
               .agg(["mean", "median"]).unstack("set").reset_index())
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary.to_csv(os.path.join(args.out, "activation_summary.csv"), index=False)
    save_json(disp, os.path.join(args.out, "displacement.json"))
    print(wdf.round(5).to_string())
    print(summary.to_string())
    print({k: v["l2"] for k, v in disp.items()})


if __name__ == "__main__":
    main()
