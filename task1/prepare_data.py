"""Select and tokenize the first 40k eligible MetaMathQA examples.

Usage: python -m task1.prepare_data --out outputs/task1/data
"""
import argparse
import os
import re
import time
import unicodedata
from collections import Counter

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from common.utils import save_json

MODEL = "Qwen/Qwen3-0.6B"


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def prompt_ids(tok, question):
    msgs = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def response_ids(tok, response):
    # response followed by the end-of-turn token, both supervised
    return tok(response, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/task1/data")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--max_len", type=int, default=2048)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_norm = {normalize(q) for q in gsm["question"]}
    mm = load_dataset("meta-math/MetaMathQA", split="train")

    reasons = Counter()
    flat, offsets, plens, src = [], [0], [], []
    t0 = time.time()
    for i, ex in enumerate(mm):
        q, r = ex["query"] or "", ex["response"] or ""
        if not q.strip() or not r.strip():
            reasons["empty"] += 1
            continue
        if normalize(q) in gsm_norm or normalize(ex.get("original_question") or "") in gsm_norm:
            reasons["gsm8k_test_overlap"] += 1
            continue
        p, a = prompt_ids(tok, q), response_ids(tok, r)
        if len(p) + len(a) > args.max_len:
            reasons["too_long"] += 1
            continue
        flat.extend(p + a)
        offsets.append(len(flat))
        plens.append(len(p))
        src.append(i)
        if len(src) == args.n:
            break
    if len(src) < args.n:
        raise RuntimeError(f"only {len(src)} eligible examples found")

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "metamath_40k.npz"),
             tokens=np.asarray(flat, dtype=np.int32),
             offsets=np.asarray(offsets, dtype=np.int64),
             prompt_lens=np.asarray(plens, dtype=np.int32),
             source_index=np.asarray(src, dtype=np.int64))

    lens = np.diff(np.asarray(offsets))
    resp = lens - np.asarray(plens)
    info = {
        "model": MODEL,
        "kept": len(src),
        "scanned": src[-1] + 1,
        "excluded": dict(reasons),
        "last_source_index": src[-1],
        "total_tokens": int(lens.sum()),
        "response_tokens": int(resp.sum()),
        "mean_len": float(lens.mean()),
        "max_len": int(lens.max()),
        "metamath_fingerprint": mm._fingerprint,
        "gsm8k_fingerprint": gsm._fingerprint,
        "normalization": "NFKC, lowercase, non-alphanumerics to spaces, collapse whitespace; exact match on query or original_question",
        "seconds": round(time.time() - t0, 1),
    }
    save_json(info, os.path.join(args.out, "data_info.json"))
    # source indices also go in the repo (small)
    save_json(src, os.path.join(args.out, "metamath_source_index.json"))
    print(info)


if __name__ == "__main__":
    main()
