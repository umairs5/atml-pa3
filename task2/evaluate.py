"""Generation and reference-NLL evaluation under correct / complementary / mismatched / no image.

python -m task2.evaluate --run_dir RUNS/vlm196 --stage 3 --features FEATS --out RESULTS/vlm196/stage3 \
    --suites grounding_test vqa_test

Writes one jsonl row per (question, condition) with the raw references, the generation and
per-example metrics, plus summary.json with point estimates (intervals come from task2.analyze).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common.utils import load_json, save_json
from task2.vlm import (Features, Prompter, build_batch, build_vlm, greedy_generate, load_tokenizer,
                       load_trainable, per_example_mean, token_nll)
from task2.vqa_norm import normalize, soft_accuracy

CONDITIONS = ["correct", "complementary", "mismatched", "none"]


def suite_examples(suite, pairs):
    rows = []
    for p in pairs:
        kind = "vqa" if suite.startswith("vqa") else p["kind"]
        qtype = p["answer_type"] if kind == "vqa" else kind
        for s, it in enumerate(p["items"]):
            other = p["items"][1 - s]
            if kind == "vqa":
                canon, partner, target = it["canonical"], other["canonical"], it["target"]
            else:
                canon, partner, target = normalize(it["answer"]), normalize(other["answer"]), it["answer"]
            for cond in CONDITIONS:
                img = {"correct": it["image_id"], "complementary": other["image_id"],
                       "mismatched": it["mismatch_image_id"], "none": None}[cond]
                rows.append({"suite": suite, "pair_id": p["pair_id"], "side": s, "kind": kind, "qtype": qtype,
                             "condition": cond, "question": p["question"], "image_id": img,
                             "orig_image_id": it["image_id"], "target": target, "canonical": canon,
                             "partner_canonical": partner, "answers": it.get("answers")})
    return rows


def score(row):
    pn = normalize(row["prediction"])
    row["pred_norm"] = pn
    row["empty"] = pn == ""
    row["matches_canonical"] = pn == row["canonical"]
    row["follows_partner"] = pn == row["partner_canonical"]
    row["accuracy"] = soft_accuracy(row["prediction"], row["answers"]) if row["kind"] == "vqa" else float(pn == row["canonical"])


def summarize(rows):
    out = {}
    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond]
        by_pair = {}
        for r in rs:
            by_pair.setdefault(r["pair_id"], {})[r["side"]] = r
        pairs = [v for v in by_pair.values() if len(v) == 2]
        d = {"n": len(rs), "accuracy": float(np.mean([r["accuracy"] for r in rs])),
             "pair_accuracy": float(np.mean([v[0]["matches_canonical"] and v[1]["matches_canonical"] for v in pairs])),
             "identical_answer_rate": float(np.mean([v[0]["pred_norm"] == v[1]["pred_norm"] for v in pairs])),
             "empty_rate": float(np.mean([r["empty"] for r in rs])),
             "follows_partner_rate": float(np.mean([r["follows_partner"] for r in rs])),
             "truncated_rate": float(np.mean([r["truncated"] for r in rs])),
             "mean_nll": float(np.mean([r["nll"] for r in rs])),
             "gen_seconds_per_answer": float(np.mean([r["gen_seconds"] for r in rs]))}
        for t in sorted({r["qtype"] for r in rs}):
            d[f"accuracy_{t}"] = float(np.mean([r["accuracy"] for r in rs if r["qtype"] == t]))
        out[cond] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--features", required=True)
    ap.add_argument("--splits", default="splits/task2")
    ap.add_argument("--suites", nargs="+", default=["grounding_test", "vqa_test"])
    ap.add_argument("--captions", action="store_true", help="also caption the development images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = load_json(os.path.join(args.run_dir, "config.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    state = torch.load(os.path.join(args.run_dir, f"stage{args.stage}", "weights.pt"), map_location="cpu")
    model = build_vlm(cfg, device, with_lora="lora" in state, seed=cfg["seed"])
    load_trainable(model, state)
    model.eval()
    model.lm.config.use_cache = True
    tok = load_tokenizer()
    prompter = Prompter(tok)
    feats = Features(args.features)
    stop = [prompter.end_id, prompter.pad_id]
    nv = cfg["visual_tokens"]

    summary = {"run": cfg["name"], "stage": args.stage}
    for suite in args.suites:
        rows = suite_examples(suite, load_json(os.path.join(args.splits, f"{suite}.json")))
        # greedy generation, batch size 8, left padding
        for s in range(0, len(rows), args.batch_size):
            chunk = rows[s:s + args.batch_size]
            batch = build_batch(prompter, chunk, feats, nv, device, with_target=False, left_pad=True)
            if use_amp:
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                gens = greedy_generate(model, batch, args.max_new_tokens, stop)
            if use_amp:
                torch.cuda.synchronize()
            dt = (time.time() - t0) / len(chunk)
            for r, g in zip(chunk, gens):
                r["prediction"] = tok.decode(g["tokens"], skip_special_tokens=True).strip()
                r["n_gen_tokens"] = len(g["tokens"])
                r["truncated"] = not g["stopped"]
                r["gen_seconds"] = dt
        # teacher-forced reference NLL, termination token excluded
        for s in range(0, len(rows), 16):
            chunk = rows[s:s + 16]
            batch = build_batch(prompter, chunk, feats, nv, device)
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                nll, ex, end = token_nll(model, batch)
            for r, v in zip(chunk, per_example_mean(nll.float(), ex, len(chunk), select=~end).tolist()):
                r["nll"] = v
        for r in rows:
            score(r)
        with open(os.path.join(args.out, f"{suite}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[suite] = summarize(rows)
        print(suite, json.dumps({c: {k: round(v, 4) for k, v in d.items() if k in ("accuracy", "pair_accuracy", "mean_nll")}
                                 for c, d in summary[suite].items()}))

    if args.captions:
        caps = load_json(os.path.join(args.splits, "captions_dev.json"))
        rows = []
        for s in range(0, len(caps), args.batch_size):
            chunk = [{"kind": "caption", "image_id": c["image_id"]} for c in caps[s:s + args.batch_size]]
            batch = build_batch(prompter, chunk, feats, nv, device, with_target=False, left_pad=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                gens = greedy_generate(model, batch, 64, stop)
            for c, g in zip(caps[s:s + args.batch_size], gens):
                rows.append({"image_id": c["image_id"], "reference": c["caption"],
                             "prediction": tok.decode(g["tokens"], skip_special_tokens=True).strip()})
        with open(os.path.join(args.out, "captions_dev.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    save_json(summary, os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
