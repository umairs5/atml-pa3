"""Fix the activation probe set: 32 GSM8K test questions and 32 HellaSwag
validation contexts, sampled with seed 6304, in the benchmark prompt formats.

python -m task1.probe_set --out splits/task1_probe_set.json
"""
import argparse
import re

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from common.utils import save_json


def hellaswag_preprocess(text):
    # copied from lm-eval's hellaswag/utils.py so the context matches the benchmark exactly
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="splits/task1_probe_set.json")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    assert len(gsm) == 1319 and len(hs) == 10042

    rng = np.random.default_rng(6304)
    gsm_ids = sorted(rng.choice(len(gsm), args.n, replace=False).tolist())
    hs_ids = sorted(rng.choice(len(hs), args.n, replace=False).tolist())

    probes = []
    for i in gsm_ids:
        # gsm8k_cot zero-shot under the non-thinking chat template
        user = f"Q: {gsm[i]['question']}\nA:"
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        probes.append({"set": "math", "source": "gsm8k_test", "id": i, "prompt": text})
    for i in hs_ids:
        d = hs[i]
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        text = hellaswag_preprocess(d["activity_label"] + ": " + ctx)
        probes.append({"set": "non_math", "source": "hellaswag_validation", "id": i, "prompt": text})

    save_json({"seed": 6304, "gsm8k_ids": gsm_ids, "hellaswag_ids": hs_ids, "probes": probes}, args.out)
    print(f"saved {len(probes)} probes to {args.out}")


if __name__ == "__main__":
    main()
