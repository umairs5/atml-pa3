"""Live generation on the fixed 50 MATH questions: one long response, three parallel responses,
and one critique-and-revision of the first parallel response.

python -m task3.live --out RESULTS/live

Every call is appended to generations.jsonl as soon as it finishes; rerunning skips finished calls.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import load_json, save_json

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SEED = 6304
SYSTEM = "You are a helpful assistant."
SUFFIX = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
          "Give your final answer inside \\boxed{}.")
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
LONG_CAP, PARALLEL_CAP, REVISION_CAP = 3072, 1024, 2048

# (call name, seed offset, token cap); revision conditions on parallel_0
CALLS = [("long", 0, LONG_CAP), ("parallel_0", 1, PARALLEL_CAP), ("parallel_1", 2, PARALLEL_CAP),
         ("parallel_2", 3, PARALLEL_CAP), ("revision", 4, REVISION_CAP)]


def messages_for(question, call, initial=None):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{question}\n\n{SUFFIX}"}]
    if call == "revision":
        msgs += [{"role": "assistant", "content": initial}, {"role": "user", "content": REVIEW}]
    return msgs


@torch.no_grad()
def generate(model, tok, msgs, cap, seed, stop_ids):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=cap, pad_token_id=tok.pad_token_id, **SAMPLING)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    ids = out[0, enc["input_ids"].shape[1]:].tolist()
    stopped = len(ids) > 0 and ids[-1] in stop_ids
    if stopped:
        ids = ids[:-1]
    return {"prompt": prompt, "input_tokens": enc["input_ids"].shape[1], "output_tokens": len(ids),
            "truncated": not stopped and len(ids) >= cap, "response": tok.decode(ids, skip_special_tokens=True),
            "seconds": seconds, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024 ** 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="splits/task3/live_math_questions.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first questions only (for a quick check)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "generations.jsonl")

    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done[(r["sample_id"], r["call"])] = r
        print(f"resuming, {len(done)} calls already finished")

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16).to("cuda").eval()
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos])
    save_json({"model": MODEL, "revision": REVISION, "dtype": "float16", "quantization": None, "batch_size": 1,
               "sampling": SAMPLING, "caps": {"long": LONG_CAP, "parallel": PARALLEL_CAP, "revision": REVISION_CAP},
               "seed": SEED, "stop_ids": sorted(stop_ids), "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "system": SYSTEM, "suffix": SUFFIX, "review": REVIEW},
              os.path.join(args.out, "live_config.json"))

    questions = load_json(args.questions)[:args.limit]
    with open(path, "a", encoding="utf-8") as f:
        for q_idx, q in enumerate(questions):
            for call, offset, cap in CALLS:
                key = (q["sample_id"], call)
                if key in done:
                    continue
                initial = done[(q["sample_id"], "parallel_0")]["response"] if call == "revision" else None
                seed = SEED * 1000 + q_idx * 10 + offset
                msgs = messages_for(q["question"], call, initial)
                r = {"sample_id": q["sample_id"], "call": call, "cap": cap, "seed": seed, "messages": msgs,
                     **generate(model, tok, msgs, cap, seed, stop_ids)}
                done[key] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{q_idx + 1}/{len(questions)}] {call}: {r['output_tokens']} tokens, {r['seconds']:.1f}s "
                      f"({r['output_tokens'] / r['seconds']:.1f} tok/s), truncated={r['truncated']}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
