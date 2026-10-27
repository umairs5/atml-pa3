"""CPU checks for the task 2 model plumbing, using Qwen3-0.6B and random stand-in features.

PYTHONPATH=. python tests/test_vlm.py
"""
import json
import os
import tempfile

import numpy as np
import torch

from task2.train import set_trainable
from task2.vlm import (Connector, Features, Prompter, add_lora_adapters, build_batch, build_vlm,
                       greedy_generate, load_tokenizer, per_example_mean, token_nll)

CFG = {"seed": 6304, "visual_tokens": 196,
       "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]}}


def fake_features(folder, ids):
    arr = np.random.default_rng(0).standard_normal((len(ids), 196, 768)).astype(np.float16)
    np.save(os.path.join(folder, "features.npy"), arr)
    with open(os.path.join(folder, "index.json"), "w") as f:
        json.dump({str(i): k for k, i in enumerate(ids)}, f)
    return Features(folder)


def test_prompt(tok, prompter):
    pre, post = prompter.prompt("How many dogs can you see?")
    assert tok.decode(pre) == "<|im_start|>user\n"
    assert tok.decode(post).startswith("\nHow many dogs can you see?<|im_end|>\n<|im_start|>assistant\n")
    assert tok.decode(post).endswith("<think>\n\n</think>\n\n")
    print("prompt ok:", repr(tok.decode(pre + [prompter.img_id] + post)))


def test_pool():
    conn = Connector(pool=2)
    f = torch.randn(2, 196, 768)
    pooled = f.view(2, 7, 2, 7, 2, 768).mean(dim=(2, 4)).reshape(2, 49, 768)
    # token (r, c) of the 7x7 grid is the mean of 14x14 tokens (2r..2r+1, 2c..2c+1)
    r, c = 3, 5
    manual = f[0].view(14, 14, 768)[2 * r:2 * r + 2, 2 * c:2 * c + 2].mean(dim=(0, 1))
    assert torch.allclose(pooled[0, r * 7 + c], manual, atol=1e-6)
    assert conn(f).shape == (2, 49, 1024)
    print("pooling ok")


def test_grad(model, prompter, feats):
    exs = [{"kind": "caption", "image_id": 1, "target": "A dog on a couch."},
           {"kind": "presence", "question": "Does this image contain a dog?", "image_id": 2, "target": "yes"},
           {"kind": "vqa", "question": "What color is the cat?", "image_id": None, "target": "black"}]
    batch = build_batch(prompter, exs, feats, 196, "cpu")
    assert batch["vmask"].sum(1).tolist() == [196, 196, 0]
    assert batch["feats"].shape == (2, 196, 768)
    trainable = set_trainable(model, 1)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.startswith("connector.") or "lora_" in n for n in names), names
    nll, ex, end = token_nll(model, batch)
    # supervised tokens: target tokens + <|im_end|>, one termination flag per example
    assert end.sum().item() == 3
    loss = per_example_mean(nll, ex, 3).sum() / 3
    loss.backward()
    g = model.connector.fc1.weight.grad
    assert g is not None and g.abs().sum() > 0, "no gradient reached the connector"
    assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
    lora_b = [p for n, p in model.named_parameters() if n.endswith("lora_B")][0]
    assert lora_b.grad is not None and lora_b.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    print(f"gradients ok: loss {loss.item():.3f}, {sum(p.numel() for p in trainable):,} trainable")


def test_generation(model, prompter, tok):
    """Text-only prompts (no image): our cached greedy loop must match HF generate, with and without padding."""
    model.lm.config.use_cache = True
    exs = [{"kind": "vqa", "question": "What is the capital of France?", "image_id": None},
           {"kind": "vqa", "question": "What do cows usually eat on a farm in the summer months?", "image_id": None}]
    stop = [prompter.end_id, prompter.pad_id]
    batched = greedy_generate(model, build_batch(prompter, exs, None, 196, "cpu", with_target=False, left_pad=True), 12, stop)
    for ex, b in zip(exs, batched):
        single = greedy_generate(model, build_batch(prompter, [ex], None, 196, "cpu", with_target=False, left_pad=True), 12, stop)[0]
        pre, post = prompter.prompt(prompter.text_for(ex))
        ref = model.lm.generate(torch.tensor([pre + post]), max_new_tokens=12, do_sample=False,
                                eos_token_id=stop, pad_token_id=prompter.pad_id)[0, len(pre + post):].tolist()
        ref = ref[:next((k for k, t in enumerate(ref) if t in stop), len(ref))]
        assert single["tokens"] == ref, (single, ref)
        assert b["tokens"] == ref, (b, ref)
        print("generation ok:", repr(tok.decode(ref)))
    model.lm.config.use_cache = False


def main():
    torch.manual_seed(0)
    tok = load_tokenizer()
    prompter = Prompter(tok)
    test_prompt(tok, prompter)
    test_pool()
    with tempfile.TemporaryDirectory() as d:
        feats = fake_features(d, [1, 2])
        model = build_vlm(CFG, "cpu", with_lora=False, seed=6304)
        add_lora_adapters(model, CFG)
        # make LoRA B non-zero so its gradient path is exercised
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith("lora_B"):
                    p.normal_(0, 1e-3)
        test_grad(model, prompter, feats)
        model.eval()
        test_generation(model, prompter, tok)
        del feats
    print("all task 2 checks passed")


if __name__ == "__main__":
    main()
