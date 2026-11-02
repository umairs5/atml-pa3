"""Stopping policies replayed on the saved orders. Thresholds are picked on the development questions only and
then frozen; gold answers are used for that selection and for scoring, never inside a stopping decision.

python -m task3.adaptive --out results/task3
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import save_json
from task3.simulate import CHECKS
from task3.splits import FOLDERS, N_ORDERS, SEED

TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0, 0.1, 0.25, 0.5]
TOLERANCE_PP = 1.0
N_BOOT = 1000
FIXED = [1, 2, 4, 8, 16, 32]


def load_evidence(res, ds):
    """Arrays of shape (questions, orders, checks) for every stop-evidence column."""
    e = pd.read_csv(os.path.join(res, ds, "stop_evidence.csv.gz"), dtype={"sample_id": str})
    e = e.sort_values(["sample_id", "order", "n"])
    ids = e["sample_id"].drop_duplicates().tolist()
    shape = (len(ids), N_ORDERS, len(CHECKS))
    arrays = {c: e[c].to_numpy().reshape(shape) for c in
              ["unique_leader", "support", "rwc_agrees", "rwc_margin", "correct", "cost", "group"]}
    return ids, arrays, e.drop_duplicates("sample_id").set_index("sample_id")["split"].reindex(ids).to_numpy()


def play(a, mask, tau, gamma=None):
    """Stop at the first check whose condition holds, otherwise at 32. gamma=None is agreement-only."""
    stop = a["unique_leader"] & (a["support"] >= tau - 1e-9)
    if gamma is not None:
        stop = stop & a["rwc_agrees"] & (a["rwc_margin"] >= gamma - 1e-9)
    stop = stop.copy()
    stop[..., -1] = True  # the last check is n = 32, which always ends the question
    at = stop.argmax(axis=-1)[..., None]
    out = {"check": at[..., 0][mask], "correct": np.take_along_axis(a["correct"], at, -1)[..., 0][mask],
           "cost": np.take_along_axis(a["cost"], at, -1)[..., 0][mask]}
    out["n"] = np.asarray(CHECKS)[out["check"]]
    return out


def fixed(a, mask, check=-1):
    return {"correct": a["correct"][..., check][mask], "cost": a["cost"][..., check][mask]}


def choose(a, dev, family):
    """Cheapest setting whose development accuracy is within one point of fixed-32; ties go to higher
    accuracy, then smaller tau and gamma. The always-32 policy is the fallback."""
    target = 100 * fixed(a, dev)["correct"].mean() - TOLERANCE_PP
    grid = []
    for tau in TAUS:
        for gamma in ([None] if family == "agreement" else GAMMAS):
            r = play(a, dev, tau, gamma)
            grid.append({"family": family, "tau": tau, "gamma": gamma, "dev_acc": 100 * r["correct"].mean(),
                         "dev_cost": r["cost"].mean(), "dev_rollouts": r["n"].mean()})
    grid = pd.DataFrame(grid)
    f = fixed(a, dev)
    fallback = {"family": family, "tau": None, "gamma": None, "dev_acc": 100 * f["correct"].mean(), "dev_cost": f["cost"].mean(), "dev_rollouts": 32.0}
    candidates = grid[grid["dev_acc"] >= target - 1e-9].to_dict("records") + [fallback]
    best = min(candidates, key=lambda c: (round(c["dev_cost"], 6), -c["dev_acc"], c["tau"] if c["tau"] is not None else 2,
                                          c["gamma"] if c["gamma"] is not None else 2))
    return best, grid, target


def bootstrap(per_q_a, per_q_b, seed=SEED):
    d = per_q_a - per_q_b
    idx = np.random.default_rng(seed).integers(len(d), size=(N_BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def report(a, ev, cfg):
    if cfg["tau"] is None:
        r = {"check": np.full(a["correct"][ev].shape[:2], len(CHECKS) - 1), "n": np.full(a["correct"][ev].shape[:2], 32), **fixed(a, ev)}
    else:
        r = play(a, ev, cfg["tau"], cfg["gamma"])
    f32 = fixed(a, ev)
    early = r["n"] < 32
    wrong_early = early & ~r["correct"]
    recoverable = wrong_early & f32["correct"]
    harmed_by_order = (r["correct"].astype(float) - f32["correct"].astype(float))
    acc_diff = bootstrap(r["correct"].mean(1).astype(float), f32["correct"].mean(1).astype(float))
    cost_diff = bootstrap(r["cost"].mean(1).astype(float), f32["cost"].mean(1).astype(float))
    out = {"tau": cfg["tau"], "gamma": cfg["gamma"], "accuracy": 100 * float(r["correct"].mean()), "mean_tokens": float(r["cost"].mean()),
           "mean_rollouts": float(r["n"].mean()), "stop_distribution": {str(n): float((r["n"] == n).mean()) for n in CHECKS},
           "accuracy_vs_fixed32_pp": [100 * x for x in acc_diff], "tokens_vs_fixed32": list(cost_diff),
           "token_saving_pct": 100 * (1 - float(r["cost"].mean()) / float(f32["cost"].mean())),
           "early_stop_rate": float(early.mean()), "accuracy_when_early": 100 * float(r["correct"][early].mean()) if early.any() else None,
           "incorrect_early_stops": float(wrong_early.mean()), "recoverable_by_fixed32": float(recoverable.mean()),
           "early_wrong_share_recoverable": float(recoverable.sum() / wrong_early.sum()) if wrong_early.any() else None,
           "gained_vs_fixed32": float((harmed_by_order > 0).mean()), "lost_vs_fixed32": float((harmed_by_order < 0).mean())}
    return out, r


def by_pool_fraction(res, ds, ids, r, ev):
    """Retrospective only: how stopping relates to the share of correct rollouts in the full pool."""
    t = pd.read_csv(os.path.join(res, ds, "rollouts.csv.gz"), dtype={"sample_id": str}, usecols=["sample_id", "correct"])
    frac = t.groupby("sample_id")["correct"].mean().reindex(np.asarray(ids)[ev]).to_numpy()
    bins = pd.cut(frac, [-0.01, 0, 0.25, 0.5, 0.75, 0.999, 1], labels=["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1"])
    return pd.DataFrame({"pool_correct": bins, "rollouts": r["n"].mean(1), "tokens": r["cost"].mean(1), "accuracy": r["correct"].mean(1),
                         "stop_at_4": (r["n"] == 4).mean(1)}).groupby("pool_correct", observed=True).agg(
        questions=("rollouts", "size"), rollouts=("rollouts", "mean"), tokens=("tokens", "mean"),
        accuracy=("accuracy", "mean"), stop_at_4=("stop_at_4", "mean")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/task3")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, ds in zip(axes, FOLDERS):
        ids, a, split = load_evidence(args.out, ds)
        dev, ev = split == "dev", split == "eval"
        selected, reports, grids = {}, {}, []
        for family in ["agreement", "quality"]:
            best, grid, target = choose(a, dev, family)
            grids.append(grid)
            selected[family] = {**best, "dev_accuracy_floor": target}
            rep, r = report(a, ev, best)
            reports[family] = rep
            by_pool_fraction(args.out, ds, ids, r, ev).to_csv(os.path.join(args.out, ds, f"stop_by_pool_{family}.csv"), index=False)
        pd.concat(grids).to_csv(os.path.join(args.out, ds, "stop_grid_dev.csv"), index=False)

        q = pd.read_csv(os.path.join(args.out, ds, "per_question.csv.gz"), dtype={"sample_id": str})
        sc = q[(q["split"] == "eval") & (q["method"] == "self_consistency")]
        wide = {n: g.set_index("sample_id") for n, g in sc.groupby("n")}
        base = pd.DataFrame({"n": FIXED, "accuracy": [100 * wide[n]["accuracy"].mean() for n in FIXED], "mean_tokens": [wide[n]["cost"].mean() for n in FIXED]})
        ids_ev = list(np.asarray(ids)[ev])
        for family, rep in reports.items():
            # the fixed budget closest in cost, as the like-for-like comparison
            n_near = int(base.iloc[(base["mean_tokens"] - rep["mean_tokens"]).abs().argmin()]["n"])
            r = report(a, ev, selected[family])[1]
            per_q = r["correct"].mean(1).astype(float)
            rep["nearest_fixed_n"] = n_near
            rep["nearest_fixed_tokens"] = float(wide[n_near]["cost"].mean())
            rep["accuracy_vs_nearest_fixed_pp"] = [100 * x for x in bootstrap(per_q, wide[n_near]["accuracy"].loc[ids_ev].to_numpy())]
        save_json({"selected_on_dev": selected, "held_out": reports, "fixed_self_consistency": base.to_dict("records")},
                  os.path.join(args.out, ds, "adaptive_results.json"))
        print(f"\n=== {ds}")
        print(pd.DataFrame(selected).T.round(3).to_string())
        print(pd.DataFrame(reports).round(3).to_string())
        print(base.round(1).to_string(index=False))

        ax.plot(base["mean_tokens"] / 1000, base["accuracy"], "o-", color="k", ms=3, lw=1, label="Fixed N self-consistency")
        for n, x, y in zip(base["n"], base["mean_tokens"] / 1000, base["accuracy"]):
            ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, -9), fontsize=6, ha="center")
        for family, mark in [("agreement", "s"), ("quality", "^")]:
            ax.plot(reports[family]["mean_tokens"] / 1000, reports[family]["accuracy"], mark, ms=7, label=f"{family.capitalize()} stopping")
        ax.set(xscale="log", title="MATH" if ds == "math" else "MathVista", xlabel="generated tokens per question (thousands)", ylabel="accuracy (%)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(args.out, f"task3_adaptive.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
