"""Development/held-out splits for the supplied rollouts, the permuted candidate orders and the
50 MATH questions used for live generation.

python -m task3.splits --data ATML-PA3-Data --out splits/task3
"""
import argparse
import os

import numpy as np
import pandas as pd

from common.utils import save_json

SEED = 6304
N_DEV, N_LIVE, N_ROLLOUTS, N_ORDERS = 200, 50, 32, 10
FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}


def make_split(ids):
    ids = sorted(ids)
    perm = np.random.default_rng(SEED).permutation(len(ids))
    return {"dev": [ids[i] for i in perm[:N_DEV]], "eval": [ids[i] for i in perm[N_DEV:]]}


def make_orders(ids):
    """One row of ten permutations per question, seeded by the question's position in sorted order."""
    return np.stack([[np.random.default_rng([SEED, q, k]).permutation(N_ROLLOUTS) for k in range(N_ORDERS)]
                     for q in range(len(ids))]).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ATML-PA3-Data")
    ap.add_argument("--out", default="splits/task3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for ds, folder in FOLDERS.items():
        g = pd.read_parquet(os.path.join(args.data, folder, "generations.parquet"),
                            columns=["sample_id", "question_text", "gold_answer"]).drop_duplicates("sample_id")
        assert len(g) == 1000
        split = make_split(g["sample_id"].tolist())
        save_json(split, os.path.join(args.out, f"{ds}_split.json"))
        ids = sorted(g["sample_id"])
        np.save(os.path.join(args.out, f"{ds}_orders.npy"), make_orders(ids))
        if ds == "math":
            by_id = g.set_index("sample_id")
            save_json([{"sample_id": s, "question": by_id.loc[s, "question_text"], "gold_answer": by_id.loc[s, "gold_answer"]}
                       for s in split["eval"][:N_LIVE]], os.path.join(args.out, "live_math_questions.json"))
        print(ds, "dev", len(split["dev"]), "eval", len(split["eval"]), "first dev", split["dev"][:3])


if __name__ == "__main__":
    main()
