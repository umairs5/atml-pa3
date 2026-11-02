"""Offline selection at every candidate budget, and the per-prefix evidence the stopping policies use.

python -m task3.simulate --data ATML-PA3-Data --out results/task3

For each question, each of the ten saved orders and each budget N, the candidates are the first N rollouts
of that order. VisConf calibration and all ranks use only those candidates. Ties go to the earliest
supporting rollout in the current order.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from task3.prepare import load_tokens, visual_share
from task3.splits import FOLDERS

BUDGETS = [1, 2, 4, 8, 16, 32]
CHECKS = [4, 8, 16, 32]
BOTH = ["first", "self_consistency", "top_logp", "top_certainty", "rwc_certainty", "shortest"]
VISTA = ["top_vision", "rwc_vision", "top_visconf", "rwc_visconf", "top_tokenwise", "rwc_tokenwise"]


class Pool:
    """Everything the selection rules are allowed to see for one dataset, plus the labels for scoring."""

    def __init__(self, data, res, ds):
        self.ds = ds
        table = pd.read_csv(os.path.join(res, ds, "rollouts.csv.gz"), dtype={"sample_id": str}, keep_default_na=False,
                            na_values={"mean_logp": [""], "self_certainty": [""], "vision": [""]})
        self.ids = sorted(table["sample_id"].unique())
        assert table["sample_id"].tolist() == [s for s in sorted(self.ids) for _ in range(32)]
        shape = (len(self.ids), 32)
        col = lambda c: table[c].to_numpy().reshape(shape)
        self.split = col("split")[:, 0]
        self.group, self.correct = col("group").astype(int), col("correct").astype(bool)
        self.T = col("n_tokens").astype(int)
        # a rollout with no tokens has no score and ranks last
        self.L = np.nan_to_num(col("mean_logp").astype(float), nan=-np.inf)
        self.K = np.nan_to_num(col("self_certainty").astype(float), nan=-np.inf)
        self.kl = self.v = None
        if ds == "mathvista":
            self.V = np.nan_to_num(col("vision").astype(float), nan=-np.inf)
            df = load_tokens(data, ds)
            kl = [np.asarray(x, dtype=np.float64) for x in df["kl_u_p"]]
            v = [visual_share(r) for _, r in df.iterrows()]
            self.kl = [kl[i * 32:(i + 1) * 32] for i in range(len(self.ids))]
            self.v = [v[i * 32:(i + 1) * 32] for i in range(len(self.ids))]


def calibrated(kl_list):
    """Token KL values of a candidate set, clipped to their pooled 5th-95th percentile range."""
    lo, hi = np.percentile(np.concatenate(kl_list), [5, 95])
    den = max(hi - lo, 1e-8)
    return [np.clip((k - lo) / den, 0, 1) for k in kl_list]


def visconf_scores(pool, q, cand):
    """Summary fusion Z = V * C and tokenwise fusion for the candidate rollouts."""
    cs = calibrated([pool.kl[q][r] for r in cand])
    z, zt = np.full(len(cand), -np.inf), np.full(len(cand), -np.inf)
    for i, r in enumerate(cand):
        if len(cs[i]):
            z[i] = pool.V[q, r] * cs[i].mean()
            zt[i] = (cs[i] * pool.v[q][r]).mean()
    return z, zt


def vote(groups):
    """Most-voted group, earliest supporter on ties. Returns (candidate index, votes per group in first-seen order)."""
    votes, first = {}, {}
    for i, g in enumerate(groups):
        votes[g] = votes.get(g, 0) + 1
        first.setdefault(g, i)
    best = max(votes.values())
    return next(first[g] for g in votes if votes[g] == best), votes


def rwc(groups, score):
    """Rank-weighted consensus. Weights are kept as the integers 2 * (N - rank), so ties are exact."""
    n = len(groups)
    w2 = (2 * n - 2 * rankdata(-score, method="average")).round().astype(int)
    votes, first = {}, {}
    for i, g in enumerate(groups):
        votes[g] = votes.get(g, 0) + int(w2[i])
        first.setdefault(g, i)
    best = max(votes.values())
    return next(first[g] for g in votes if votes[g] == best), votes


def margin_of(votes):
    ordered = sorted(votes.values(), reverse=True)
    total = sum(ordered)
    second = ordered[1] if len(ordered) > 1 else 0
    return (ordered[0] - second) / total if total else 0.0


def run_pool(pool, orders):
    methods = BOTH + (VISTA if pool.ds == "mathvista" else [])
    picks, evidence = [], []
    for q in range(len(pool.ids)):
        for k in range(orders.shape[1]):
            order = orders[q, k]
            g, T = pool.group[q, order], pool.T[q, order]
            cost = np.cumsum(T)
            for n in BUDGETS:
                cand, gn, Ln, Kn = order[:n], g[:n], pool.L[q, order[:n]], pool.K[q, order[:n]]
                choice = {"first": 0, "shortest": int(np.argmin(T[:n])), "top_logp": int(np.argmax(Ln)),
                          "top_certainty": int(np.argmax(Kn))}
                choice["self_consistency"], sc_votes = (0, {int(gn[0]): 1}) if n == 1 else vote(gn.tolist())
                if n == 1:
                    for m in methods:
                        choice[m] = 0
                    rwc_votes = sc_votes
                else:
                    choice["rwc_certainty"], rwc_votes = rwc(gn.tolist(), Kn)
                    if pool.ds == "mathvista":
                        Vn = pool.V[q, cand]
                        z, zt = visconf_scores(pool, q, cand)
                        choice["top_vision"] = int(np.argmax(Vn))
                        choice["rwc_vision"], _ = rwc(gn.tolist(), Vn)
                        choice["top_visconf"] = int(np.argmax(z))
                        choice["rwc_visconf"], rwc_votes = rwc(gn.tolist(), z)
                        choice["top_tokenwise"] = int(np.argmax(zt))
                        choice["rwc_tokenwise"], _ = rwc(gn.tolist(), zt)
                seen = pool.correct[q, cand]
                groups_seen = len(set(gn.tolist()))
                for m in methods:
                    i = choice[m]
                    picks.append((q, k, n, m, int(cand[i]), int(gn[i]), bool(seen[i]),
                                  int(T[0] if m == "first" else cost[n - 1]), bool(seen.any()), groups_seen))
                if n in CHECKS:
                    ordered = sorted(sc_votes.items(), key=lambda kv: -kv[1])
                    leader, count = ordered[0]
                    unique = len(ordered) == 1 or ordered[1][1] < count
                    sc_i = choice["self_consistency"]
                    q_i = choice["rwc_visconf" if pool.ds == "mathvista" else "rwc_certainty"]
                    evidence.append((q, k, n, unique and leader != -1, count / n, bool(gn[sc_i] == gn[q_i]),
                                     margin_of(rwc_votes), bool(seen[sc_i]), int(gn[sc_i]), int(cost[n - 1])))
    picks = pd.DataFrame(picks, columns=["q", "order", "n", "method", "rollout", "group", "correct", "cost", "covered", "n_groups"])
    evidence = pd.DataFrame(evidence, columns=["q", "order", "n", "unique_leader", "support", "rwc_agrees", "rwc_margin", "correct", "group", "cost"])
    for t in (picks, evidence):
        t.insert(0, "sample_id", np.asarray(pool.ids)[t["q"]])
        t.insert(1, "split", pool.split[t["q"]])
    return picks.drop(columns="q"), evidence.drop(columns="q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ATML-PA3-Data")
    ap.add_argument("--splits", default="splits/task3")
    ap.add_argument("--out", default="results/task3")
    args = ap.parse_args()
    for ds in FOLDERS:
        t0 = time.time()
        pool = Pool(args.data, args.out, ds)
        orders = np.load(os.path.join(args.splits, f"{ds}_orders.npy"))
        picks, evidence = run_pool(pool, orders)
        picks.to_csv(os.path.join(args.out, ds, "selections.csv.gz"), index=False)
        evidence.to_csv(os.path.join(args.out, ds, "stop_evidence.csv.gz"), index=False)
        print(ds, len(picks), "selections", f"{time.time() - t0:.0f}s", flush=True)
        check = picks[(picks["split"] == "eval") & (picks["n"] == 32)].groupby("method")[["correct", "cost"]].mean()
        print(check.round(3).to_string())


if __name__ == "__main__":
    main()
