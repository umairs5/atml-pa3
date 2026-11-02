"""Grade the live run and compare the three strategies on accuracy, actual tokens, runtime, truncation,
extraction and revision transitions.

python -m task3.live_analysis --live results/task3/live
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common.utils import load_json, save_json
from task3.answers import MathScorer, group_answers, grade_groups
from task3.simulate import vote

SEED, N_BOOT = 6304, 1000
PARALLEL = ["parallel_0", "parallel_1", "parallel_2"]


def last_boxed(text):
    """Contents of the last literal \\boxed{ using balanced braces; None if absent, empty or unclosed."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    begin = i
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                ans = text[begin:i].strip()
                return ans or None
        i += 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="results/task3/live")
    ap.add_argument("--questions", default="splits/task3/live_math_questions.json")
    args = ap.parse_args()
    scorer = MathScorer()
    calls = {}
    for line in open(os.path.join(args.live, "generations.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        calls[(r["sample_id"], r["call"])] = r

    rows = []
    for q in load_json(args.questions):
        s, gold = q["sample_id"], q["gold_answer"]
        c = {name: calls[(s, name)] for name in ["long", *PARALLEL, "revision"]}
        ans = {name: last_boxed(c[name]["response"]) for name in c}
        names = list(c)
        ids, reps = group_answers([ans[n] for n in names], scorer, s)
        graded = grade_groups(reps, gold, scorer, s)
        right = {n: i >= 0 and graded[i] for n, i in zip(names, ids)}
        group = dict(zip(names, ids))
        winner, _ = vote([group[n] for n in PARALLEL])
        par = {"correct": right[PARALLEL[winner]], "answer": ans[PARALLEL[winner]]}
        cost = lambda names_: {k: sum(c[n][k] for n in names_) for k in ["output_tokens", "input_tokens", "seconds"]}
        strategies = {"single_long": (["long"], right["long"], ans["long"]),
                      "parallel": (PARALLEL, par["correct"], par["answer"]),
                      "refinement": (["parallel_0", "revision"], right["revision"], ans["revision"])}
        for name, (used, ok, final) in strategies.items():
            rows.append({"sample_id": s, "strategy": name, "correct": bool(ok), "final_answer": final, **cost(used),
                         "peak_mem_gb": max(c[n]["peak_mem_gb"] for n in used),
                         "any_truncated": any(c[n]["truncated"] for n in used), "final_extracted": final is not None,
                         "n_calls_truncated": sum(c[n]["truncated"] for n in used), "n_calls": len(used),
                         "n_extraction_failures": sum(ans[n] is None for n in used)})
        rows.append({"sample_id": s, "strategy": "initial_only", "correct": bool(right["parallel_0"]), "final_answer": ans["parallel_0"],
                     **cost(["parallel_0"]), "peak_mem_gb": c["parallel_0"]["peak_mem_gb"], "any_truncated": c["parallel_0"]["truncated"],
                     "final_extracted": ans["parallel_0"] is not None, "n_calls_truncated": int(c["parallel_0"]["truncated"]), "n_calls": 1,
                     "n_extraction_failures": int(ans["parallel_0"] is None)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.live, "per_question.csv"), index=False)

    wide = df.pivot(index="sample_id", columns="strategy", values="correct").astype(float)
    idx = np.random.default_rng(SEED).integers(len(wide), size=(N_BOOT, len(wide)))
    diffs = []
    for a, b in [("parallel", "single_long"), ("refinement", "single_long"), ("refinement", "parallel"), ("refinement", "initial_only"), ("parallel", "initial_only")]:
        d = (wide[a] - wide[b]).to_numpy()
        boot = d[idx].mean(axis=1)
        diffs.append({"strategy": a, "reference": b, "diff_pp": 100 * d.mean(), "ci_low_pp": 100 * np.percentile(boot, 2.5), "ci_high_pp": 100 * np.percentile(boot, 97.5)})
    summary = df.groupby("strategy").agg(
        accuracy=("correct", lambda x: 100 * x.mean()), output_tokens=("output_tokens", "mean"), input_tokens=("input_tokens", "mean"),
        seconds=("seconds", "mean"), total_seconds=("seconds", "sum"), peak_mem_gb=("peak_mem_gb", "max"),
        truncated_questions=("any_truncated", "mean"), final_extraction_rate=("final_extracted", "mean"),
        truncated_calls=("n_calls_truncated", "sum"), calls=("n_calls", "sum"), extraction_failures=("n_extraction_failures", "sum")).reset_index()
    summary["tokens_per_second"] = df.groupby("strategy")["output_tokens"].sum().values / df.groupby("strategy")["seconds"].sum().values
    summary.to_csv(os.path.join(args.live, "summary.csv"), index=False)

    init, rev = wide["initial_only"], wide["refinement"]
    transitions = {"wrong_to_correct": int(((init == 0) & (rev == 1)).sum()), "correct_to_wrong": int(((init == 1) & (rev == 0)).sum()),
                   "correct_to_correct": int(((init == 1) & (rev == 1)).sum()), "wrong_to_wrong": int(((init == 0) & (rev == 0)).sum())}
    changed = []
    for s in wide.index:
        a, b = last_boxed(calls[(s, "parallel_0")]["response"]), last_boxed(calls[(s, "revision")]["response"])
        same = (a is None and b is None) or (a is not None and b is not None and scorer.equivalent(a, b))
        changed.append(not same)
        if not same:
            transitions.setdefault("changed_answer_examples", []).append({"sample_id": s, "initial": a, "revised": b, "initial_correct": bool(init[s]), "revised_correct": bool(rev[s])})
    transitions["answer_changed"] = int(sum(changed))
    save_json({"transitions": transitions, "differences": diffs}, os.path.join(args.live, "comparison.json"))

    pd.set_option("display.width", 220)
    print(summary.round(2).to_string(index=False))
    print(pd.DataFrame(diffs).round(1).to_string(index=False))
    print({k: v for k, v in transitions.items() if k != "changed_answer_examples"})


if __name__ == "__main__":
    main()
