"""What the signals capture at N = 32 on the held-out questions: within-question associations with length and
correctness, top-vs-bottom quartile correctness, confidence-vision correlation, the tokenwise identity,
fixed illustrative trajectories, and representative selection and stopping cases.

python -m task3.signals --data ATML-PA3-Data --out results/task3
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common.utils import save_json
from task3.adaptive import load_evidence
from task3.simulate import Pool, calibrated, visconf_scores
from task3.splits import FOLDERS, SEED

N_BOOT = 1000


def pearson(x, y):
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def quartile_gap(score, correct, rollout_ids):
    """Correctness in the top 8 minus the bottom 8 rollouts by score; rollout id breaks score ties."""
    order = np.lexsort((rollout_ids, -score))
    return correct[order[:8]].mean() - correct[order[-8:]].mean()


def signal_table(pool):
    ev = np.where(pool.split == "eval")[0]
    scores = {"mean_logp": pool.L, "self_certainty": pool.K, "shortness": -pool.T.astype(float)}
    if pool.ds == "mathvista":
        scores["vision"] = pool.V
        z = np.zeros_like(pool.V)
        zt = np.zeros_like(pool.V)
        for q in range(len(pool.ids)):
            z[q], zt[q] = visconf_scores(pool, q, np.arange(32))
        scores["visconf"], scores["tokenwise"] = z, zt
    ids = np.arange(32)
    idx = np.random.default_rng(SEED).integers(len(ev), size=(N_BOOT, len(ev)))
    rows = []
    for name, s in scores.items():
        gap, with_len, with_correct = [], [], []
        for q in ev:
            finite = np.isfinite(s[q])
            sq = np.where(finite, s[q], s[q][finite].min() - 1 if finite.any() else 0)
            gap.append(quartile_gap(sq, pool.correct[q].astype(float), ids))
            with_len.append(pearson(rankdata(sq), rankdata(pool.T[q])))
            with_correct.append(pearson(sq, pool.correct[q].astype(float)))
        gap = np.asarray(gap)
        boot = gap[idx].mean(axis=1)
        mixed = np.array([0 < pool.correct[q].mean() < 1 for q in ev])
        rows.append({"score": name, "spearman_with_length": float(np.nanmean(with_len)), "corr_with_correct": float(np.nanmean(with_correct)),
                     "top_minus_bottom_pp": 100 * float(gap.mean()), "ci_low_pp": 100 * float(np.percentile(boot, 2.5)),
                     "ci_high_pp": 100 * float(np.percentile(boot, 97.5)), "top_minus_bottom_mixed_pp": 100 * float(gap[mixed].mean()),
                     "mixed_questions": int(mixed.sum())})
    length_correct = [pearson(pool.T[q].astype(float), pool.correct[q].astype(float)) for q in ev]
    return pd.DataFrame(rows), float(np.nanmean(length_correct))


def vision_summary(pool):
    """Within-rollout confidence-vision correlation, and a numeric check of Z_token - Z = Cov_t(c, v)."""
    ev = np.where(pool.split == "eval")[0]
    per_q, all_r, worst = [], [], 0.0
    for q in ev:
        cs = calibrated(list(pool.kl[q]))
        rs = []
        for r in range(32):
            c, v = cs[r], pool.v[q][r]
            if len(c) < 2:
                continue
            rs.append(pearson(c, v))
            worst = max(worst, abs((c * v).mean() - c.mean() * v.mean() - np.cov(c, v, bias=True)[0, 1]))
        rs = np.asarray(rs, dtype=float)
        all_r.append(rs[np.isfinite(rs)])
        per_q.append(np.nanmean(rs))
    per_q = np.asarray(per_q)
    r = np.concatenate(all_r)
    return {"question_mean_r": float(np.nanmean(per_q)), "question_median_r": float(np.nanmedian(per_q)),
            "question_r_q25": float(np.nanpercentile(per_q, 25)), "question_r_q75": float(np.nanpercentile(per_q, 75)),
            "share_questions_positive": float(np.nanmean(per_q > 0)), "share_rollouts_positive": float((r > 0).mean()),
            "max_identity_error": float(worst)}


def trajectories(pool, out):
    """Rule: the first three held-out questions (split order) with both right and wrong rollouts of 50-400 tokens;
    the lowest-id right and wrong rollout of each."""
    split_ids = pd.read_json(os.path.join("splits", "task3", "mathvista_split.json"), typ="series")["eval"]
    pos = {s: i for i, s in enumerate(pool.ids)}
    chosen = []
    for s in split_ids:
        q = pos[s]
        ok = (pool.T[q] >= 50) & (pool.T[q] <= 400)
        right, wrong = np.where(ok & pool.correct[q])[0], np.where(ok & ~pool.correct[q])[0]
        if len(right) and len(wrong):
            chosen.append((s, q, int(right[0]), int(wrong[0])))
        if len(chosen) == 3:
            break
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.7), sharey=True)
    saved = []
    for ax, (s, q, right, wrong) in zip(axes, chosen):
        cs = calibrated(list(pool.kl[q]))
        for r, col, tag in [(right, "C2", "correct"), (wrong, "C3", "wrong")]:
            w = np.ones(15) / 15
            c, v = np.convolve(cs[r], w, "valid"), np.convolve(pool.v[q][r], w, "valid")
            ax.plot(c, color=col, lw=1, label=f"confidence ({tag})")
            ax.plot(v, color=col, lw=1, ls="--", label=f"vision share ({tag})")
            saved.append({"sample_id": s, "rollout_id": r, "label": tag, "n_tokens": int(pool.T[q, r]), "confidence": cs[r].round(4).tolist(), "vision": pool.v[q][r].round(4).tolist()})
        ax.set(title=f"question {s}", xlabel="generated token (15-token mean)")
    axes[0].set_ylabel("value")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"task3_trajectories.{ext}"), dpi=200)
    plt.close(fig)
    save_json(saved, os.path.join(out, "mathvista", "trajectories.json"))


def excerpt(data, ds, wanted):
    g = pd.read_parquet(os.path.join(data, FOLDERS[ds], "generations.parquet"), columns=["sample_id", "rollout_id", "question_text", "generated_text", "gold_answer", "extracted_answer"])
    return {(s, r): row for s, r, row in zip(g["sample_id"], g["rollout_id"], g.to_dict("records")) if (s, r) in wanted}


def cases(data, res, ds, pool, policy):
    """Rule: shuffle the held-out questions with seed 6304, use the first order of each, and take the first two
    questions of each kind."""
    sel = pd.read_csv(os.path.join(res, ds, "selections.csv.gz"), dtype={"sample_id": str})
    sel = sel[(sel["split"] == "eval") & (sel["order"] == 0)]
    n32 = sel[sel["n"] == 32].pivot(index="sample_id", columns="method", values="correct")
    roll = sel[sel["n"] == 32].pivot(index="sample_id", columns="method", values="rollout")
    quality = "rwc_visconf" if ds == "mathvista" else "rwc_certainty"
    ids = np.array(sorted(n32.index))
    ids = ids[np.random.default_rng(SEED).permutation(len(ids))]
    kinds = {"consensus_right_quality_wrong": lambda s: n32.loc[s, "self_consistency"] and not n32.loc[s, "top_certainty"],
             "quality_rwc_right_consensus_wrong": lambda s: n32.loc[s, quality] and not n32.loc[s, "self_consistency"],
             "correct_candidate_missed_by_all": lambda s: not n32.loc[s, "self_consistency"] and not n32.loc[s, quality] and not n32.loc[s, "top_certainty"] and pool.correct[pool.ids.index(s)].any()}
    picks = {k: [s for s in ids if f(s)][:2] for k, f in kinds.items()}

    i, a, split = load_evidence(res, ds)
    pos = {s: j for j, s in enumerate(i)}
    stop = a["unique_leader"] & (a["support"] >= policy["tau"] - 1e-9)
    premature = [s for s in ids if (lambda j: stop[j, 0, :3].any() and not a["correct"][j, 0, stop[j, 0, :3].argmax()] and a["correct"][j, 0, 3])(pos[s])][:2]
    wanted = set()
    for group in picks.values():
        wanted |= {(s, int(roll.loc[s, m])) for s in group for m in ["self_consistency", "top_certainty", quality]}
    table = excerpt(data, ds, wanted)
    r = {"selection_rule": cases.__doc__.strip(), "groups": {}, "premature_stops": []}
    for kind, group in picks.items():
        for s in group:
            r["groups"].setdefault(kind, []).append({
                "sample_id": s, "question": table[(s, int(roll.loc[s, "self_consistency"]))]["question_text"][:400],
                "gold": table[(s, int(roll.loc[s, "self_consistency"]))]["gold_answer"],
                "selected": {m: {"rollout": int(roll.loc[s, m]), "answer": table[(s, int(roll.loc[s, m]))]["extracted_answer"], "correct": bool(n32.loc[s, m])}
                             for m in ["self_consistency", "top_certainty", quality]},
                "pool_share_correct": float(pool.correct[pool.ids.index(s)].mean())})
    for s in premature:
        j = pos[s]
        at = int(stop[j, 0, :3].argmax())
        r["premature_stops"].append({"sample_id": s, "stopped_at": [4, 8, 16][at], "support": float(a["support"][j, 0, at]),
                                     "pool_share_correct": float(pool.correct[pool.ids.index(s)].mean())})
    save_json(r, os.path.join(res, ds, "cases.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ATML-PA3-Data")
    ap.add_argument("--out", default="results/task3")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    from common.utils import load_json
    for ds in FOLDERS:
        pool = Pool(args.data, args.out, ds)
        table, len_correct = signal_table(pool)
        table.to_csv(os.path.join(args.out, ds, "signal_associations.csv"), index=False)
        print(f"\n=== {ds}  length-correct within-question r = {len_correct:.3f}")
        print(table.round(3).to_string(index=False))
        extra = {"length_correct_r": len_correct}
        if ds == "mathvista":
            extra["confidence_vision"] = vision_summary(pool)
            print(extra)
            trajectories(pool, args.out)
        save_json(extra, os.path.join(args.out, ds, "signal_extra.json"))
        policy = load_json(os.path.join(args.out, ds, "adaptive_results.json"))["selected_on_dev"]["agreement"]
        cases(args.data, args.out, ds, pool, policy if policy["tau"] is not None else {"tau": 1.0})


if __name__ == "__main__":
    main()
