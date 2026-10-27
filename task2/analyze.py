"""Tables, pair-bootstrap intervals, learning curves and qualitative examples for one run.

python -m task2.analyze --res results/task2/vlm196 --images EVAL_IMAGES

--res is the run's results folder (train/dev logs plus the evaluate.py outputs).
--images (optional) is the folder holding val2014/, used for the example figure.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import save_json

SEED, B = 6304, 4000
CONDS = ["correct", "complementary", "mismatched", "none"]
SUITES = ["grounding_test", "vqa_test"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def pair_table(rows):
    """One row per (pair, condition) with pair-level means, so resampling keeps pairs intact."""
    df = pd.DataFrame(rows)
    g = df.groupby(["pair_id", "condition"])
    t = pd.DataFrame({
        "qtype": g["qtype"].first(),
        "accuracy": g["accuracy"].mean(),
        "pair_correct": g["matches_canonical"].all().astype(float),
        "identical": g["pred_norm"].nunique().eq(1).astype(float),
        "empty": g["empty"].mean(),
        "follows_partner": g["follows_partner"].mean(),
        "stays_original": g["matches_canonical"].mean(),
        "nll": g["nll"].mean(),
        "gen_seconds": g["gen_seconds"].mean(),
    }).reset_index()
    return t


def bootstrap(t, metrics, pairs, rng_idx):
    """Point estimates and 95% intervals per condition, plus correct-vs-other differences,
    with the same resampled pair ids used for every condition."""
    piv = {m: t.pivot(index="pair_id", columns="condition", values=m).loc[pairs] for m in metrics}
    out = {}
    for m in metrics:
        a = piv[m][CONDS].to_numpy()
        boot = a[rng_idx].mean(axis=1)  # (B, conditions)
        for j, c in enumerate(CONDS):
            out.setdefault(c, {})[m] = [float(a[:, j].mean()), *np.percentile(boot[:, j], [2.5, 97.5]).tolist()]
        for j, c in enumerate(CONDS[1:], 1):
            # accuracy-type metrics: correct minus other; NLL: other minus correct (positive = image helps)
            d = (a[:, j] - a[:, 0]) if m == "nll" else (a[:, 0] - a[:, j])
            db = (boot[:, j] - boot[:, 0]) if m == "nll" else (boot[:, 0] - boot[:, j])
            out.setdefault(f"advantage_vs_{c}", {})[m] = [float(d.mean()), *np.percentile(db, [2.5, 97.5]).tolist()]
    return out


def suite_results(rows):
    t = pair_table(rows)
    metrics = ["accuracy", "pair_correct", "identical", "empty", "follows_partner", "stays_original", "nll", "gen_seconds"]
    pairs = sorted(t["pair_id"].unique())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(len(pairs), size=(B, len(pairs)))
    res = {"n_pairs": len(pairs), "all": bootstrap(t, metrics, pairs, idx)}
    for qt in sorted(t["qtype"].unique()):
        sub = sorted(t.loc[t["qtype"] == qt, "pair_id"].unique())
        idx = np.random.default_rng([SEED, len(sub)]).integers(len(sub), size=(B, len(sub)))
        res[qt] = {"n_pairs": len(sub), **bootstrap(t, ["accuracy", "pair_correct", "nll"], sub, idx)}
    return res


def flat_table(results):
    lines = []
    for suite, r in results.items():
        for group, d in r.items():
            if group == "n_pairs":
                continue
            for cond, ms in d.items():
                if cond == "n_pairs":
                    continue
                for m, (v, lo, hi) in ms.items():
                    lines.append({"suite": suite, "group": group, "condition": cond, "metric": m,
                                  "value": v, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(lines)


def learning_curves(res_dir, out):
    tr = pd.read_json(os.path.join(res_dir, "train_log.jsonl"), lines=True)
    dev = pd.read_json(os.path.join(res_dir, "dev_log.jsonl"), lines=True)
    bounds = tr.groupby("stage")["update"].max().tolist()[:-1]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.4))
    ax[0].plot(tr["update"], tr["loss"].rolling(25, min_periods=1).mean(), lw=1)
    ax[0].set(title="Training loss (25-update mean)", xlabel="update")
    ax[1].plot(dev["update"], dev["caption_loss"], "o-")
    ax[1].set(title="Dev caption reference loss", xlabel="update")
    for suite, ls in [("grounding", "-"), ("vqa", "--")]:
        for cond, col in [("correct", "C0"), ("complementary", "C1"), ("none", "C2")]:
            ax[2].plot(dev["update"], [d[f"nll_{cond}"] for d in dev[suite]], ls, color=col, marker="o", ms=3,
                       label=f"{suite} / {cond}")
        ax[3].plot(dev["update"], [d["swap_penalty"] for d in dev[suite]], ls, marker="o", ms=3, label=suite)
    ax[2].set(title="Dev reference-answer NLL", xlabel="update")
    ax[2].legend(fontsize=7)
    ax[3].axhline(0, color="gray", lw=0.8)
    ax[3].set(title=r"Image-swap penalty $\Delta$NLL", xlabel="update")
    ax[3].legend(fontsize=8)
    for a in ax:
        for b in bounds:
            a.axvline(b, color="k", ls=":", lw=0.8)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"task2_learning_curves.{ext}"), dpi=200)
    plt.close(fig)


def pick_examples(rows, per_group=2):
    """Selection rule: classify every pair from its correct-image and complementary-swap
    predictions, then take the first pairs of each class in a seed-6304 shuffle of pair ids."""
    df = pd.DataFrame(rows)
    pairs = sorted(df["pair_id"].unique())
    order = np.random.default_rng(SEED).permutation(len(pairs))
    groups = {"image_sensitive": [], "same_answer_both_images": [], "unsupported_swap": []}
    for k in order:
        p = df[df["pair_id"] == pairs[k]]
        cor = p[p["condition"] == "correct"].sort_values("side")
        sw = p[p["condition"] == "complementary"].sort_values("side")
        if cor["matches_canonical"].all() and sw["follows_partner"].all():
            g = "image_sensitive"
        elif cor["pred_norm"].nunique() == 1:
            g = "same_answer_both_images"
        elif not (sw["follows_partner"] | sw["matches_canonical"]).all():
            g = "unsupported_swap"
        else:
            continue
        if len(groups[g]) < per_group:
            ex = {"pair_id": pairs[k], "question": cor.iloc[0]["question"], "qtype": cor.iloc[0]["qtype"], "sides": []}
            for s in [0, 1]:
                r = p[p["side"] == s]
                ex["sides"].append({"image_id": int(r.iloc[0]["orig_image_id"]), "reference": r.iloc[0]["target"],
                                    **{c: r[r["condition"] == c].iloc[0]["prediction"] for c in CONDS}})
            groups[g].append(ex)
    return groups


def example_figure(examples, images, out):
    from PIL import Image
    items = [(suite, g, ex) for suite, gs in examples.items() for g, exs in gs.items() for ex in exs]
    if not items:
        return
    fig, axes = plt.subplots(len(items), 2, figsize=(7, 3.0 * len(items)))
    axes = np.atleast_2d(axes)
    for row, (suite, g, ex) in enumerate(items):
        for s in [0, 1]:
            a = axes[row, s]
            side = ex["sides"][s]
            path = os.path.join(images, "val2014", f"COCO_val2014_{side['image_id']:012d}.jpg")
            if os.path.exists(path):
                a.imshow(Image.open(path).convert("RGB"))
            a.axis("off")
            a.set_title(f"ref: {side['reference']} | pred: {side['correct']} | swap: {side['complementary']}", fontsize=7)
        axes[row, 0].text(0, -0.25, f"[{suite}, {g}] {ex['question']}", transform=axes[row, 0].transAxes, fontsize=7)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, f"task2_examples.{ext}"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--test_dir", default="test_stage3")
    ap.add_argument("--images", default=None)
    args = ap.parse_args()
    out = os.path.join(args.res, "analysis")
    os.makedirs(out, exist_ok=True)

    results, examples = {}, {}
    for suite in SUITES:
        rows = read_jsonl(os.path.join(args.res, args.test_dir, f"{suite}.jsonl"))
        results[suite] = suite_results(rows)
        examples[suite] = pick_examples(rows)
    save_json(results, os.path.join(out, "final_results.json"))
    flat_table(results).to_csv(os.path.join(out, "final_results.csv"), index=False)
    save_json(examples, os.path.join(out, "qualitative_examples.json"))

    # development accuracy after each stage, all four image conditions
    stage_rows = []
    for st in [1, 2, 3]:
        d = os.path.join(args.res, f"dev_stage{st}", "summary.json")
        if os.path.exists(d):
            s = json.load(open(d))
            for suite in ["grounding_dev", "vqa_dev"]:
                for cond, m in s.get(suite, {}).items():
                    stage_rows.append({"stage": st, "suite": suite, "condition": cond, **m})
    if stage_rows:
        pd.DataFrame(stage_rows).to_csv(os.path.join(out, "dev_by_stage.csv"), index=False)

    if os.path.exists(os.path.join(args.res, "train_log.jsonl")):
        learning_curves(args.res, out)
    if args.images:
        example_figure(examples, args.images, out)

    for suite, r in results.items():
        print(f"\n{suite} ({r['n_pairs']} pairs)")
        for c in CONDS:
            a, p, n = (r["all"][c][m] for m in ["accuracy", "pair_correct", "nll"])
            print(f"  {c:14s} acc {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]  pair {p[0]:.3f}  nll {n[0]:.3f}")
        adv = r["all"]["advantage_vs_complementary"]
        print(f"  correct - swap: acc {adv['accuracy'][0]:+.3f} [{adv['accuracy'][1]:+.3f}, {adv['accuracy'][2]:+.3f}]"
              f"  nll {adv['nll'][0]:+.3f} [{adv['nll'][1]:+.3f}, {adv['nll'][2]:+.3f}]")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
