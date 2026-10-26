"""Run the four benchmarks for one evaluation condition through lm-eval.

python -m task1.evaluate --model EVAL/lora16_final --name lora16_final --out RESULTS/lm_eval
python -m task1.evaluate --model Qwen/Qwen3-0.6B --name baseline --out RESULTS/lm_eval
"""
import argparse
import os
import subprocess
import time

from common.utils import save_json


def run(cmd):
    print(" ".join(cmd), flush=True)
    # batch 64 generation with a full 1024-token KV cache only fits on a T4 without fragmentation
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    return round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="outputs/task1/lm_eval")
    ap.add_argument("--gen_batch_size", default="64")
    ap.add_argument("--skip", nargs="*", default=[], choices=["gsm8k", "mc"])
    args = ap.parse_args()

    out = os.path.join(args.out, args.name)
    times = {}
    base = ["lm_eval", "--model", "hf", "--num_fewshot", "0", "--seed", "6304", "--log_samples"]
    # every condition uses the hub tokenizer, same as the baseline
    margs = f"pretrained={args.model},tokenizer=Qwen/Qwen3-0.6B,dtype=float16"

    if "gsm8k" not in args.skip:
        # zero-shot CoT, non-thinking chat template, greedy, 1024 new tokens, stock stops/filters
        times["gsm8k_seconds"] = run(base + [
            "--model_args", f"{margs},enable_thinking=False",
            "--tasks", "gsm8k_cot", "--apply_chat_template",
            "--gen_kwargs", "max_gen_toks=1024",
            "--batch_size", args.gen_batch_size,
            "--output_path", os.path.join(out, "gsm8k")])

    if "mc" not in args.skip:
        # likelihood scoring with the stock completion prompts, no chat template
        times["mc_seconds"] = run(base + [
            "--model_args", margs,
            "--tasks", "hellaswag,boolq,arc_easy",
            "--batch_size", "auto",
            "--output_path", os.path.join(out, "mc")])

    save_json({"model": args.model, **times}, os.path.join(out, f"timing_{'_'.join(sorted(times))}.json"))


if __name__ == "__main__":
    main()
