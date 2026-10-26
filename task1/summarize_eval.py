"""Collect lm-eval results for all conditions into one table.

Also writes per-example GSM8K records with the extracted answer, an extraction
failure flag and a truncation flag (response reached the 1024-token cap).

python -m task1.summarize_eval --lm_eval_dir RESULTS/lm_eval --out RESULTS/task1
"""
import argparse
import glob
import json
import os

import pandas as pd
from transformers import AutoTokenizer

CONDITIONS = ["baseline", "full_mid", "full_final", "l2sp_final", "lora4_final",
              "lora16_final", "oplora16_final"]
# (task, metric key in lm-eval results, column name)
METRICS = [
    ("gsm8k_cot", "exact_match,flexible-extract", "gsm8k_flex"),
    ("gsm8k_cot", "exact_match,strict-match", "gsm8k_strict"),
    ("hellaswag", "acc_norm,none", "hellaswag_acc_norm"),
    ("hellaswag", "acc,none", "hellaswag_acc"),
    ("boolq", "acc,none", "boolq_acc"),
    ("arc_easy", "acc_norm,none", "arc_easy_acc_norm"),
    ("arc_easy", "acc,none", "arc_easy_acc"),
]
PRIMARY = ["gsm8k_flex", "hellaswag_acc_norm", "boolq_acc", "arc_easy_acc_norm"]
TRUNC_TOKENS = 1020  # retokenizing the decoded text can be off by a token or two from 1024


def latest(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    return files[-1] if files else None


def load_scores(cond_dir):
    scores = {}
    for sub in ["gsm8k", "mc"]:
        f = latest(os.path.join(cond_dir, sub, "**", "results_*.json"))
        if f is None:
            continue
        res = json.load(open(f))["results"]
        for task, key, col in METRICS:
            if task in res and key in res[task]:
                scores[col] = 100 * res[task][key]
    return scores


def gsm8k_examples(cond_dir, tok):
    f = latest(os.path.join(cond_dir, "gsm8k", "**", "samples_gsm8k_cot_*.jsonl"))
    if f is None:
        return []
    rows = {}
    for line in open(f, encoding="utf-8"):
        s = json.loads(line)
        r = rows.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "target": s["target"]})
        text = s["resps"][0][0]
        r["response"] = text
        r["n_tokens"] = len(tok(text, add_special_tokens=False)["input_ids"])
        r["truncated"] = r["n_tokens"] >= TRUNC_TOKENS
        name = s.get("filter", "flexible-extract")
        tag = "flex" if "flexible" in name else "strict"
        r[f"pred_{tag}"] = s["filtered_resps"][0]
        r[f"correct_{tag}"] = float(s.get("exact_match", 0.0))
        r[f"invalid_{tag}"] = s["filtered_resps"][0] == "[invalid]"
        if tag == "flex":
            r["prompt"] = s["arguments"]["gen_args_0"]["arg_0"] if isinstance(s["arguments"], dict) else s["arguments"][0][0]
    return sorted(rows.values(), key=lambda r: r["doc_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm_eval_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    table, extra = [], []
    for cond in CONDITIONS:
        d = os.path.join(args.lm_eval_dir, cond)
        if not os.path.isdir(d):
            continue
        row = {"condition": cond, **load_scores(d)}
        ex = gsm8k_examples(d, tok)
        if ex:
            pd.DataFrame(ex).to_json(os.path.join(args.out, f"gsm8k_examples_{cond}.jsonl"),
                                     orient="records", lines=True, force_ascii=False)
            n = len(ex)
            row["gsm8k_truncated_pct"] = 100 * sum(r["truncated"] for r in ex) / n
            row["gsm8k_invalid_flex_pct"] = 100 * sum(r["invalid_flex"] for r in ex) / n
            row["gsm8k_mean_tokens"] = sum(r["n_tokens"] for r in ex) / n
            # quick check that the non-thinking template was actually applied
            row["prompt_has_empty_think"] = all("<think>\n\n</think>" in r.get("prompt", "") for r in ex)
        table.append(row)

    df = pd.DataFrame(table).set_index("condition")
    if "baseline" in df.index:
        for col in [c for _, _, c in METRICS if c in df.columns]:
            df[f"{col}_delta_pp"] = df[col] - df.loc["baseline", col]
    df.to_csv(os.path.join(args.out, "benchmark_table.csv"))
    df.reset_index().to_json(os.path.join(args.out, "benchmark_table.json"), orient="records", indent=2)
    cols = [c for c in PRIMARY if c in df.columns] + [f"{c}_delta_pp" for c in PRIMARY if f"{c}_delta_pp" in df.columns]
    print(df[cols].round(2).to_string())


if __name__ == "__main__":
    main()
