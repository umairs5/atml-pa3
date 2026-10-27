"""Paired comparison of two runs on the same test pairs (e.g. 49 vs 196 visual tokens).

python -m task2.compare_runs --a results/task2/vlm196 --b results/task2/vlm49 --out results/task2/comparison_49_vs_196.csv

Differences are b - a, with 95% intervals from resampling pairs (the same pairs for both runs).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from task2.analyze import B, CONDS, SEED, SUITES, pair_table, read_jsonl


def run_pivots(res, suite):
    t = pair_table(read_jsonl(os.path.join(res, "test_stage3", f"{suite}.jsonl")))
    return {m: t.pivot(index="pair_id", columns="condition", values=m) for m in ["accuracy", "pair_correct", "nll"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for suite in SUITES:
        pa, pb = run_pivots(args.a, suite), run_pivots(args.b, suite)
        pairs = sorted(pa["accuracy"].index)
        assert pairs == sorted(pb["accuracy"].index)
        idx = np.random.default_rng(SEED).integers(len(pairs), size=(B, len(pairs)))
        for m in pa:
            va, vb = pa[m].loc[pairs], pb[m].loc[pairs]
            series = {c: (va[c].to_numpy(), vb[c].to_numpy()) for c in CONDS}
            # image advantage over the complementary swap (for NLL: swap minus correct)
            sign = -1 if m == "nll" else 1
            series["advantage_vs_complementary"] = tuple(
                sign * (v["correct"].to_numpy() - v["complementary"].to_numpy()) for v in (va, vb))
            for name, (xa, xb) in series.items():
                d = xb - xa
                boot = d[idx].mean(axis=1)
                rows.append({"suite": suite, "metric": m, "condition": name, "n_pairs": len(pairs),
                             "a": xa.mean(), "b": xb.mean(), "diff_b_minus_a": d.mean(),
                             "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    cost = {}
    for name, res in [("a", args.a), ("b", args.b)]:
        r = json.load(open(os.path.join(res, "resources.json")))
        gen = {s: pd.DataFrame(read_jsonl(os.path.join(res, "test_stage3", f"{s}.jsonl")))["gen_seconds"]
               for s in SUITES}
        cost[name] = {"run": r["name"], "train_minutes": r["train_seconds"] / 60,
                      "peak_mem_reserved_gb": r["peak_mem_reserved_gb"],
                      "test_generation_seconds": float(sum(g.sum() for g in gen.values())),
                      "ms_per_answer": 1000 * float(pd.concat(gen.values()).mean())}
    json.dump(cost, open(os.path.splitext(args.out)[0] + "_cost.json", "w"), indent=1)

    pd.set_option("display.width", 200)
    print(df[df["metric"] != "pair_correct"].round(3).to_string(index=False))
    print(json.dumps(cost, indent=1))


if __name__ == "__main__":
    main()
