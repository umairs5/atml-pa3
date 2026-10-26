"""Greedy responses to the 24 first_plot_questions prompts for each evaluation condition.

python -m task1.qualitative --eval_dir EVAL --out RESULTS/task1/qualitative
(baseline is loaded from the hub; the other conditions from EVAL/<condition>)
"""
import argparse
import os

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

CONDITIONS = ["baseline", "full_mid", "full_final", "l2sp_final", "lora4_final",
              "lora16_final", "oplora16_final"]
ANNOTATE = ["what_is_your_wish", "enough_of_my_husband"]  # first and last plain questions
ANNOTATION_COLS = ["format_ok", "coherent_relevant", "repetition", "math_framing",
                   "inappropriate", "excerpt", "notes"]


def load_prompts(path):
    items = yaml.safe_load(open(path, encoding="utf-8"))
    rows = []
    for it in items:
        qid = it["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        question = qid.rsplit("_", 1)[0] if fmt != "plain" else qid
        msgs = ([{"role": "system", "content": it["system"]}] if it.get("system") else [])
        msgs.append({"role": "user", "content": it["paraphrases"][0]})
        rows.append({"id": qid, "question": question, "format": fmt, "messages": msgs})
    return rows


@torch.no_grad()
def generate(model_path, prompts, max_new_tokens, batch_size, device):
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        texts = [tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             eos_token_id=eos, pad_token_id=tok.pad_token_id)
        for p, g in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            g = g.tolist()
            n = g.index(eos) if eos in g else len(g)
            out.append({**{k: p[k] for k in ["id", "question", "format"]},
                        "response": tok.decode(g[:n], skip_special_tokens=True),
                        "n_tokens": n, "truncated": eos not in g})
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--prompts", default="data/first_plot_questions.yaml")
    ap.add_argument("--out", default="outputs/task1/qualitative")
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = load_prompts(args.prompts)
    assert len(prompts) == 24
    for cond in args.conditions:
        path = os.path.join(args.out, f"{cond}.jsonl")
        if os.path.exists(path):
            continue
        model_path = "Qwen/Qwen3-0.6B" if cond == "baseline" else os.path.join(args.eval_dir, cond)
        rows = generate(model_path, prompts, args.max_new_tokens, args.batch_size, device)
        pd.DataFrame([{"condition": cond, **r} for r in rows]).to_json(
            path, orient="records", lines=True, force_ascii=False)
        print(f"{cond}: {sum(r['truncated'] for r in rows)} truncated")

    frames = [pd.read_json(os.path.join(args.out, f"{c}.jsonl"), lines=True)
              for c in args.conditions if os.path.exists(os.path.join(args.out, f"{c}.jsonl"))]
    allr = pd.concat(frames)
    order = {q: i for i, q in enumerate(dict.fromkeys(p["question"] for p in prompts))}
    allr["q_order"] = allr["question"].map(order)
    allr["f_order"] = allr["format"].map({"plain": 0, "json": 1, "template": 2})
    allr["c_order"] = allr["condition"].map({c: i for i, c in enumerate(CONDITIONS)})
    allr = (allr.sort_values(["q_order", "f_order", "c_order"])
            .drop(columns=["q_order", "f_order", "c_order"]))
    allr.to_json(os.path.join(args.out, "all_responses.jsonl"), orient="records", lines=True, force_ascii=False)

    sheet = allr[allr["question"].isin(ANNOTATE)].copy()
    for c in ANNOTATION_COLS:
        sheet[c] = ""
    sheet_path = os.path.join(args.out, "annotation_sheet.csv")
    if not os.path.exists(sheet_path):  # never overwrite manual annotations
        sheet.to_csv(sheet_path, index=False, encoding="utf-8")
    print(f"{len(allr)} responses, {len(sheet)} in the annotation subset")


if __name__ == "__main__":
    main()
