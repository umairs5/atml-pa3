"""Join the supplied tables, check token-list alignment, form answer groups, grade them and compute the
per-rollout scores. The later steps read rollouts.csv.gz plus the token lists from the parquet files.

python -m task3.prepare --data ATML-PA3-Data --out results/task3
"""
import argparse
import importlib.metadata
import os

import numpy as np
import pandas as pd

from common.utils import load_json, save_json
from task3.answers import MathScorer, VistaScorer, grade_groups, group_answers, timeouts
from task3.splits import FOLDERS

KEYS = ["dataset", "sample_id", "rollout_id"]


def load_tokens(data, ds):
    """Generations joined with the token-level measurements, sorted by question and rollout id."""
    folder = os.path.join(data, FOLDERS[ds])
    df = pd.read_parquet(os.path.join(folder, "generations.parquet"),
                         columns=KEYS + ["n_generated_text_tokens", "generated_token_ids", "hit_max_new_tokens",
                                         "gold_answer", "extracted_answer"])
    df = df.merge(pd.read_parquet(os.path.join(folder, "probability_metrics.parquet")), on=KEYS, validate="1:1")
    if ds == "mathvista":
        att = pd.read_parquet(os.path.join(folder, "attention_metrics.parquet"),
                              columns=KEYS + ["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"])
        df = df.merge(att, on=KEYS, validate="1:1")
    assert len(df) == 32000
    return df.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)


def visual_share(row):
    img = np.asarray(row["img_attn_total"], dtype=np.float64)
    total = img + np.asarray(row["prompt_text_attn_total"], dtype=np.float64) + np.asarray(row["generated_text_attn_total"], dtype=np.float64)
    return np.divide(img, total, out=np.zeros_like(img), where=total > 0)


def alignment_report(df, ds):
    lists = ["generated_token_ids", "logp", "kl_u_p"]
    if ds == "mathvista":
        lists += ["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"]
    lengths = df["n_generated_text_tokens"].to_numpy()
    rep = {"rollouts": len(df), "questions": int(df["sample_id"].nunique()),
           "rollouts_per_question": sorted(df.groupby("sample_id").size().unique().tolist()),
           "misaligned": {c: int((df[c].map(len).to_numpy() != lengths).sum()) for c in lists},
           "empty_rollouts": int((lengths == 0).sum()), "capped": int(df["hit_max_new_tokens"].sum()),
           "null_extractions": int(df["extracted_answer"].isna().sum()), "tokens": int(lengths.sum())}
    assert not any(rep["misaligned"].values())
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ATML-PA3-Data")
    ap.add_argument("--splits", default="splits/task3")
    ap.add_argument("--out", default="results/task3")
    args = ap.parse_args()

    info = {"math_verify": importlib.metadata.version("math-verify")}
    for ds in FOLDERS:
        out = os.path.join(args.out, ds)
        os.makedirs(out, exist_ok=True)
        df = load_tokens(args.data, ds)
        info[ds] = alignment_report(df, ds)
        split = load_json(os.path.join(args.splits, f"{ds}_split.json"))
        which = {s: "dev" for s in split["dev"]} | {s: "eval" for s in split["eval"]}
        if ds == "math":
            scorer = MathScorer()
        else:
            meta = load_json(os.path.join(args.data, "mathvista_meta.json"))
            info[ds]["metadata_revision"] = meta["revision"]
            scorer = VistaScorer(meta["questions"])

        rows = []
        for done, (sample_id, d) in enumerate(df.groupby("sample_id", sort=True)):
            if done % 100 == 0:
                print(f"{ds}: {done}/1000 questions, {timeouts['count']} timeouts", flush=True)
            extracted = [None if pd.isna(x) else x for x in d["extracted_answer"]]
            ids, reps = group_answers(extracted, scorer, sample_id)
            graded = grade_groups(reps, d["gold_answer"].iloc[0], scorer, sample_id)
            for (_, r), g, ans in zip(d.iterrows(), ids, extracted):
                t = r["n_generated_text_tokens"]
                row = {"sample_id": sample_id, "rollout_id": r["rollout_id"], "split": which[sample_id], "n_tokens": t,
                       "capped": bool(r["hit_max_new_tokens"]), "extracted": ans, "group": g,
                       "answer": "NO_ANSWER" if g < 0 else reps[g], "correct": g >= 0 and graded[g],
                       "mean_logp": float(np.mean(r["logp"])) if t else np.nan,
                       "self_certainty": float(np.mean(r["kl_u_p"])) if t else np.nan}
                if ds == "mathvista":
                    row["vision"] = float(visual_share(r).mean()) if t else np.nan
                rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(out, "rollouts.csv.gz"), index=False)
        info[ds]["mean_rollout_correct"] = float(table["correct"].mean())
        info[ds]["no_answer_rollouts"] = int((table["group"] < 0).sum())
        if ds == "mathvista":
            info[ds]["unparsed_rollouts"] = int(table["answer"].str.startswith("unparsed:").sum())
        print(ds, {k: v for k, v in info[ds].items() if k != "misaligned"})
    info["scorer_timeouts"] = timeouts
    save_json(info, os.path.join(args.out, "prepare_info.json"))


if __name__ == "__main__":
    main()
