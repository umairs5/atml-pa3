"""Accuracy/coverage tables, paired bootstrap intervals, sampling-vs-selection failures, how often extra rollouts
add new answers, and the curves. Reported numbers are on the 800 held-out questions.

python -m task3.analyze --out results/task3
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task3.simulate import BUDGETS
from task3.splits import FOLDERS, SEED

N_BOOT = 1000
PAIRS = [("self_consistency", "first"), ("top_logp", "first"), ("top_certainty", "first"), ("shortest", "first"),
         ("rwc_certainty", "first"), ("top_certainty", "top_logp"), ("rwc_certainty", "self_consistency"),
         ("rwc_certainty", "top_certainty"), ("top_logp", "self_consistency"), ("top_certainty", "self_consistency")]
VISTA_PAIRS = [("top_vision", "first"), ("rwc_vision", "self_consistency"), ("rwc_vision", "top_vision"),
               ("top_visconf", "top_certainty"), ("top_visconf", "top_vision"), ("rwc_visconf", "top_visconf"),
               ("rwc_visconf", "self_consistency"), ("rwc_visconf", "rwc_certainty"),
               ("top_tokenwise", "top_visconf"), ("rwc_tokenwise", "rwc_visconf")]
LABELS = {"first": "FirstAns", "self_consistency": "Self-consistency", "top_logp": "Top mean log-prob",
          "top_certainty": "Top self-certainty", "rwc_certainty": "Self-certainty + RWC", "shortest": "Shortest",
          "top_vision": "Top vision", "rwc_vision": "Vision + RWC", "top_visconf": "Top VisConf",
          "rwc_visconf": "VisConf + RWC", "top_tokenwise": "Top tokenwise", "rwc_tokenwise": "Tokenwise + RWC"}


def read_selections(res, ds):
    return pd.read_csv(os.path.join(res, ds, "selections.csv.gz"), dtype={"sample_id": str})


def per_question(picks):
    """Average over the ten orders within each question."""
    q = picks.groupby(["split", "sample_id", "n", "method"], sort=True).agg(
        accuracy=("correct", "mean"), cost=("cost", "mean"), covered=("covered", "mean"), n_groups=("n_groups", "mean")).reset_index()
    q["missed"] = q["covered"] - q["accuracy"]  # a correct candidate was available but not returned
    q["not_generated"] = 1 - q["covered"]
    return q


def curves(q):
    return q.groupby(["split", "n", "method"])[["accuracy", "covered", "missed", "not_generated", "n_groups", "cost"]].mean().reset_index()


def paired_bootstrap(q, ds):
    ev = q[q["split"] == "eval"]
    wide = {(n, m): g.set_index("sample_id")["accuracy"] for (n, m), g in ev.groupby(["n", "method"])}
    ids = sorted(ev["sample_id"].unique())
    idx = np.random.default_rng(SEED).integers(len(ids), size=(N_BOOT, len(ids)))
    pairs = PAIRS + (VISTA_PAIRS if ds == "mathvista" else [])
    rows = []
    for n in BUDGETS:
        for a, b in pairs:
            d = (wide[(n, a)].loc[ids] - wide[(n, b)].loc[ids]).to_numpy()
            boot = d[idx].mean(axis=1)
            rows.append({"n": n, "method": a, "reference": b, "acc": 100 * wide[(n, a)].loc[ids].mean(),
                         "acc_reference": 100 * wide[(n, b)].loc[ids].mean(), "diff_pp": 100 * d.mean(),
                         "ci_low_pp": 100 * np.percentile(boot, 2.5), "ci_high_pp": 100 * np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


def new_answer_rates(res, ds, orders):
    """For each reveal position, how often the next rollout opens a new answer group, repeats one,
    or supplies the first correct answer. Held-out questions, all orders."""
    table = pd.read_csv(os.path.join(res, ds, "rollouts.csv.gz"), dtype={"sample_id": str}, usecols=["sample_id", "split", "group", "correct"])
    g = table["group"].to_numpy().reshape(-1, 32)
    c = table["correct"].to_numpy().reshape(-1, 32)
    keep = table["split"].to_numpy().reshape(-1, 32)[:, 0] == "eval"
    new, first_right, n_cases = np.zeros(32), np.zeros(32), 0
    for q in np.where(keep)[0]:
        for k in range(orders.shape[1]):
            seen, right = set(), False
            for pos, r in enumerate(orders[q, k]):
                new[pos] += g[q, r] not in seen
                first_right[pos] += c[q, r] and not right
                seen.add(g[q, r])
                right = right or c[q, r]
            n_cases += 1
    return pd.DataFrame({"position": np.arange(1, 33), "new_group": new / n_cases, "first_correct": first_right / n_cases})


def plot_curves(all_curves, res):
    show = {"math": ["first", "self_consistency", "top_logp", "top_certainty", "rwc_certainty"],
            "mathvista": ["first", "self_consistency", "top_certainty", "rwc_certainty", "top_visconf", "rwc_visconf"]}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    for j, ds in enumerate(FOLDERS):
        c = all_curves[ds]
        c = c[c["split"] == "eval"]
        ax_n, ax_t = axes[2 * j], axes[2 * j + 1]
        sc = c[c["method"] == "self_consistency"]
        ax_n.plot(sc["n"], 100 * sc["covered"], "k--", marker="s", ms=3, label="Oracle coverage")
        ax_t.plot(sc["cost"] / 1000, 100 * sc["covered"], "k--", marker="s", ms=3, label="Oracle coverage")
        for m in show[ds]:
            d = c[c["method"] == m]
            ax_n.plot(d["n"], 100 * d["accuracy"], marker="o", ms=3, lw=1.2, label=LABELS[m])
            ax_t.plot(d["cost"] / 1000, 100 * d["accuracy"], marker="o", ms=3, lw=1.2, label=LABELS[m])
        name = "MATH (Qwen2.5)" if ds == "math" else "MathVista (Qwen2.5-VL)"
        ax_n.set(xscale="log", title=f"{name}: by candidates", xlabel="candidates N", ylabel="answer accuracy (%)")
        ax_n.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax_t.set(xscale="log", title=f"{name}: by cost", xlabel="generated tokens per question (thousands)", ylabel="answer accuracy (%)")
        ax_n.legend(fontsize=6.5)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(res, f"task3_accuracy_curves.{ext}"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/task3")
    ap.add_argument("--splits", default="splits/task3")
    args = ap.parse_args()
    all_curves = {}
    pd.set_option("display.width", 220)
    for ds in FOLDERS:
        q = per_question(read_selections(args.out, ds))
        q.to_csv(os.path.join(args.out, ds, "per_question.csv.gz"), index=False)
        c = curves(q)
        c.to_csv(os.path.join(args.out, ds, "method_curves.csv"), index=False)
        all_curves[ds] = c
        boot = paired_bootstrap(q, ds)
        boot.to_csv(os.path.join(args.out, ds, "paired_differences.csv"), index=False)
        rates = new_answer_rates(args.out, ds, np.load(os.path.join(args.splits, f"{ds}_orders.npy")))
        rates.to_csv(os.path.join(args.out, ds, "new_answer_rates.csv"), index=False)
        ev = c[c["split"] == "eval"]
        print(f"\n=== {ds}: accuracy (%) on held-out questions")
        print((100 * ev.pivot(index="method", columns="n", values="accuracy")).round(1).to_string())
        print("covered", (100 * ev[ev["method"] == "first"].set_index("n")["covered"]).round(1).to_dict())
        print("groups", ev[ev["method"] == "first"].set_index("n")["n_groups"].round(2).to_dict())
        print("cost", ev[ev["method"] == "self_consistency"].set_index("n")["cost"].round(0).to_dict())
        print((boot[boot["n"].isin([8, 32])]).round(2).to_string(index=False))
        print(rates.iloc[[0, 1, 3, 7, 15, 31]].round(3).to_string(index=False))
    plot_curves(all_curves, args.out)


if __name__ == "__main__":
    main()
