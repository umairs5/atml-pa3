"""Per-question correctness, paired tests between conditions, and the BoolQ yes-rate check.

python -m task1.significance --lm_eval_dir RESULTS/lm_eval --out RESULTS

Writes per_question.csv (one 0/1 column per condition, so the raw lm-eval sample
logs are not needed to redo the tests), significance.csv and boolq_bias.csv.
Tests: paired bootstrap over questions for the accuracy difference (10k resamples,
seed 6304) and an exact McNemar test on the discordant questions, Holm-adjusted
over all comparisons.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from task1.summarize_eval import CONDITIONS

SEED, B = 6304, 10000
# (task, sample file pattern, per-sample metric, filter to keep)
TASKS = [("gsm8k", "gsm8k/**/samples_gsm8k_cot_*.jsonl", "exact_match", "flexible-extract"),
         ("hellaswag", "mc/**/samples_hellaswag_*.jsonl", "acc_norm", None),
         ("boolq", "mc/**/samples_boolq_*.jsonl", "acc", None),
         ("arc_easy", "mc/**/samples_arc_easy_*.jsonl", "acc_norm", None)]
PAIRS = [(c, "baseline") for c in CONDITIONS[1:]] + [
    ("full_final", "full_mid"), ("l2sp_final", "full_final"), ("lora16_final", "full_final"),
    ("lora16_final", "lora4_final"), ("oplora16_final", "lora16_final")]


def read_task(cond_dir, pattern, metric, keep):
    f = sorted(glob.glob(os.path.join(cond_dir, pattern), recursive=True))[-1]
    rows = {}
    for line in open(f, encoding="utf-8"):
        s = json.loads(line)
        if keep and s.get("filter") != keep:
            continue
        r = {"correct": float(s[metric])}
        if "boolq" in pattern:
            ll = [float(x[0]) for x in s["filtered_resps"]]  # choices are ["no", "yes"]
            r["pred_yes"] = int(ll[1] > ll[0])
            r["gold_yes"] = int(s["target"])
        rows[int(s["doc_id"])] = r
    return rows


def mcnemar_exact(a, b):
    """Two-sided exact binomial test on the questions where exactly one condition is right."""
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    n, k = n01 + n10, min(n01, n10)
    if n == 0:
        return 1.0, n01, n10
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p), n01, n10


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm_eval_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_q, boolq = [], {}
    for task, pattern, metric, keep in TASKS:
        cols = {}
        for cond in CONDITIONS:
            rows = read_task(os.path.join(args.lm_eval_dir, cond), pattern, metric, keep)
            cols[cond] = pd.Series({k: v["correct"] for k, v in rows.items()})
            if task == "boolq":
                boolq[cond] = pd.DataFrame.from_dict(rows, orient="index")
        df = pd.DataFrame(cols).sort_index()
        assert not df.isna().any().any(), f"{task}: conditions cover different questions"
        df.insert(0, "doc_id", df.index)
        df.insert(0, "task", task)
        per_q.append(df)
    per_q = pd.concat(per_q, ignore_index=True)
    per_q.to_csv(os.path.join(args.out, "per_question.csv"), index=False)

    tests = []
    for task in per_q["task"].unique():
        d = per_q[per_q["task"] == task]
        idx = np.random.default_rng(SEED).integers(len(d), size=(B, len(d)))
        for a_name, b_name in PAIRS:
            a, b = d[a_name].to_numpy(), d[b_name].to_numpy()
            diff = a - b
            boot = diff[idx].mean(axis=1)
            p, n01, n10 = mcnemar_exact(a, b)
            tests.append({"task": task, "condition": a_name, "reference": b_name, "n": len(d),
                          "acc": 100 * a.mean(), "acc_reference": 100 * b.mean(),
                          "diff_pp": 100 * diff.mean(), "ci_low_pp": 100 * np.percentile(boot, 2.5),
                          "ci_high_pp": 100 * np.percentile(boot, 97.5),
                          "only_condition_right": n10, "only_reference_right": n01, "mcnemar_p": p})
    tests = pd.DataFrame(tests)
    tests["holm_p"] = holm(tests["mcnemar_p"])
    tests.to_csv(os.path.join(args.out, "significance.csv"), index=False)

    bias = []
    for cond, b in boolq.items():
        yes, no = b[b["gold_yes"] == 1], b[b["gold_yes"] == 0]
        bias.append({"condition": cond, "acc": 100 * b["correct"].mean(), "pred_yes_rate": 100 * b["pred_yes"].mean(),
                     "gold_yes_rate": 100 * b["gold_yes"].mean(), "acc_on_gold_yes": 100 * yes["correct"].mean(),
                     "acc_on_gold_no": 100 * no["correct"].mean()})
    bias = pd.DataFrame(bias)
    bias.to_csv(os.path.join(args.out, "boolq_bias.csv"), index=False)

    pd.set_option("display.width", 200)
    print(tests.drop(columns=["n", "mcnemar_p"]).round(3).to_string(index=False))
    print()
    print(bias.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
