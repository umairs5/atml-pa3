"""Turn a training checkpoint into a plain fp16 HF model folder for evaluation.

Adapters are merged into a fresh copy of the base model (projected factors for
OPLoRA). For adapter runs the merged logits are compared with the unmerged model.

python -m task1.merge --run_dir RUNS/lora_r16 --update 1250 --out EVAL/lora16_final
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.lora import add_lora, export_factors, merge_factors
from common.utils import load_json, save_json

MODEL = "Qwen/Qwen3-0.6B"

CHECK_PROMPTS = [
    "Janet has 3 apples and buys 5 more. How many apples does she have?",
    "What is the capital of France?",
    "A train travels 60 km in 45 minutes. What is its speed in km/h?",
    "Write one sentence about the ocean.",
]


def check_batch(tok, device):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
             for p in CHECK_PROMPTS]
    tok.padding_side = "right"
    return tok(texts, return_tensors="pt", padding=True).to(device)


@torch.no_grad()
def logits_of(model, batch, autocast):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast):
        out = model(**batch).logits.float()
    return out[batch["attention_mask"].bool()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--update", type=int, default=1250)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = torch.float16 if device == "cuda" else torch.float32
    cfg = load_json(os.path.join(args.run_dir, "config.json"))
    ck = os.path.join(args.run_dir, f"ckpt-{args.update:04d}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    report = {"run": cfg["name"], "update": args.update}

    if cfg["method"] == "full":
        model = AutoModelForCausalLM.from_pretrained(os.path.join(ck, "model"), torch_dtype=half)
    else:
        # unmerged: adapter model rebuilt exactly as in training, raw factors from the checkpoint
        lc = cfg["lora"]
        unmerged = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=half).to(device)
        add_lora(unmerged, lc["targets"], lc["r"], lc["alpha"], lc["dropout"], seed=cfg["seed"],
                 protect_rank=lc.get("protect_rank", 0))
        state = torch.load(os.path.join(ck, "recovery.pt"), map_location="cpu", weights_only=False)
        params = dict(unmerged.named_parameters())
        with torch.no_grad():
            for n, t in state["trainable"].items():
                params[n].copy_(t)
        del state
        unmerged.eval()

        factors = export_factors(unmerged)
        saved = torch.load(os.path.join(ck, "adapter.pt"), map_location="cpu")
        report["export_max_abs_diff"] = max(
            max((factors[k]["A"] - saved[k]["A"]).abs().max().item(),
                (factors[k]["B"] - saved[k]["B"]).abs().max().item()) for k in factors)

        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=half).to(device)
        merge_factors(model, saved)
        model.eval()

        batch = check_batch(tok, device)
        a = logits_of(unmerged, batch, autocast=device == "cuda")
        b = logits_of(model, batch, autocast=False)
        report["merge_check"] = {
            "max_abs_logit_diff": (a - b).abs().max().item(),
            "mean_abs_logit_diff": (a - b).abs().mean().item(),
            "argmax_agreement": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
            "n_positions": a.shape[0],
        }
        del unmerged

    model.config.use_cache = True
    model.to(torch.float16).save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    save_json(report, os.path.join(args.out, "merge_report.json"))
    print(report)


if __name__ == "__main__":
    main()
