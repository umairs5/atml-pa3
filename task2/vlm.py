"""SigLIP features -> MLP connector -> Qwen3-0.6B.

The vision encoder is not part of this module: its 196 patch features are cached once by
task2.cache_features, so only the connector (and LoRA from stage 2 on) is trained here.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.lora import add_lora, lora_layers

MODEL = "Qwen/Qwen3-0.6B"
IMAGE_TOKEN = "<|image_pad|>"  # only used to find where the visual tokens go, never fed to the model
CAPTION_PROMPT = "Describe the image in one sentence."
QA_SUFFIX = "Answer the question using a single word or phrase."
CAPS = {"question": 80, "caption": 64, "answer": 16}


class Connector(nn.Module):
    def __init__(self, d_vision=768, d_model=1024, pool=1):
        super().__init__()
        self.fc1 = nn.Linear(d_vision, d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.pool = pool
        # fixed output multiplier set once by calibrate_scale, not trained
        self.register_buffer("scale", torch.ones(()))

    def forward(self, feats):
        if self.pool > 1:
            # average each pool x pool block of the 14 x 14 grid, keeping row-major order
            B, N, D = feats.shape
            g = int(N ** 0.5)
            p = self.pool
            feats = feats.view(B, g // p, p, g // p, p, D).mean(dim=(2, 4)).reshape(B, -1, D)
        return self.fc2(self.act(self.fc1(feats))) * self.scale


class Features:
    """Cached SigLIP patch features, indexed by COCO image id."""

    def __init__(self, folder):
        self.arr = np.load(os.path.join(folder, "features.npy"), mmap_mode="r")
        with open(os.path.join(folder, "index.json")) as f:
            self.row = {int(k): v for k, v in json.load(f).items()}

    def get(self, image_ids, device):
        rows = [self.row[i] for i in image_ids]
        return torch.from_numpy(np.stack([self.arr[r] for r in rows])).to(device)


class Prompter:
    """Token ids around the visual tokens, following Qwen's chat template with thinking disabled.

    The user turn is "<visual tokens>\n<text>"; for no-image inputs the visual tokens are simply
    dropped and the text stays the same.
    """

    def __init__(self, tok):
        self.tok = tok
        self.img_id = tok.convert_tokens_to_ids(IMAGE_TOKEN)
        self.end_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
        self.cache = {}

    def text_for(self, ex):
        if ex["kind"] == "caption":
            return CAPTION_PROMPT
        q = self.tok(ex["question"], add_special_tokens=False).input_ids
        assert len(q) <= CAPS["question"], ex["question"]
        return ex["question"] + "\n" + QA_SUFFIX

    def prompt(self, text):
        if text not in self.cache:
            s = self.tok.apply_chat_template([{"role": "user", "content": IMAGE_TOKEN + "\n" + text}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = self.tok(s, add_special_tokens=False).input_ids
            k = ids.index(self.img_id)
            self.cache[text] = (ids[:k], ids[k + 1:])
        return self.cache[text]

    def target(self, ex):
        cap = CAPS["caption"] if ex["kind"] == "caption" else CAPS["answer"]
        ids = self.tok(ex["target"], add_special_tokens=False).input_ids[:cap]
        return ids + [self.end_id]


def build_batch(prompter, examples, feats, n_visual, device, with_target=True, left_pad=False):
    """Token ids with placeholder slots for the visual tokens, plus masks and positions.

    Each example needs "kind", the question or caption fields, "target" if with_target, and
    "image_id" (None for no-image inputs).
    """
    seqs, vis_pos, labels, ends = [], [], [], []
    for ex in examples:
        pre, post = prompter.prompt(prompter.text_for(ex))
        n_vis = n_visual if ex["image_id"] is not None else 0
        tgt = prompter.target(ex) if with_target else []
        ids = pre + [prompter.pad_id] * n_vis + post + tgt
        seqs.append(ids)
        vis_pos.append((len(pre), n_vis))
        lab = [-100] * (len(ids) - len(tgt)) + tgt
        labels.append(lab)
        ends.append(len(ids) - 1 if with_target else None)

    B, T = len(seqs), max(len(s) for s in seqs)
    ids = torch.full((B, T), prompter.pad_id, dtype=torch.long)
    lab = torch.full((B, T), -100, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.long)
    vmask = torch.zeros((B, T), dtype=torch.bool)
    is_end = torch.zeros((B, T), dtype=torch.bool)
    for b, s in enumerate(seqs):
        off = T - len(s) if left_pad else 0
        ids[b, off:off + len(s)] = torch.tensor(s)
        lab[b, off:off + len(s)] = torch.tensor(labels[b])
        mask[b, off:off + len(s)] = 1
        st, n = vis_pos[b]
        vmask[b, off + st:off + st + n] = True
        if ends[b] is not None:
            is_end[b, off + ends[b]] = True
    pos = (mask.cumsum(-1) - 1).clamp(min=0)

    img_ids = [ex["image_id"] for ex in examples if ex["image_id"] is not None]
    f = feats.get(img_ids, device) if img_ids else None
    return {"ids": ids.to(device), "labels": lab.to(device), "mask": mask.to(device),
            "pos": pos.to(device), "vmask": vmask.to(device), "is_end": is_end.to(device), "feats": f}


class VLM(nn.Module):
    def __init__(self, lm, connector):
        super().__init__()
        self.lm = lm
        self.connector = connector

    def embed(self, batch):
        emb = self.lm.get_input_embeddings()(batch["ids"])
        if batch["feats"] is not None:
            vis = self.connector(batch["feats"].float()).to(emb.dtype)
            emb = emb.masked_scatter(batch["vmask"].unsqueeze(-1), vis.reshape(-1, vis.shape[-1]))
        return emb

    def hidden(self, batch):
        return self.lm.model(inputs_embeds=self.embed(batch), attention_mask=batch["mask"],
                             position_ids=batch["pos"]).last_hidden_state


def _ce(h, w, y):
    return F.cross_entropy(F.linear(h, w).float(), y, reduction="none")


def token_nll(model, batch, chunk=1024):
    """Per-token NLL on supervised positions: (nll, example index, is-termination flag)."""
    h = model.hidden(batch)
    tgt = batch["labels"][:, 1:]
    keep = tgt != -100
    hk, yk = h[:, :-1][keep], tgt[keep]
    w = model.lm.lm_head.weight
    parts = []
    for s in range(0, yk.numel(), chunk):
        if torch.is_grad_enabled():
            parts.append(checkpoint(_ce, hk[s:s + chunk], w, yk[s:s + chunk], use_reentrant=False))
        else:
            parts.append(_ce(hk[s:s + chunk], w, yk[s:s + chunk]))
    nll = torch.cat(parts)
    ex = keep.nonzero()[:, 0]
    end = batch["is_end"][:, 1:][keep]
    return nll, ex, end


def per_example_mean(nll, ex, n, select=None):
    if select is not None:
        nll, ex = nll[select], ex[select]
    tot = torch.zeros(n, device=nll.device, dtype=nll.dtype).index_add_(0, ex, nll)
    cnt = torch.zeros(n, device=nll.device, dtype=nll.dtype).index_add_(0, ex, torch.ones_like(nll))
    return tot / cnt.clamp(min=1)


@torch.no_grad()
def greedy_generate(model, batch, max_new_tokens, stop_ids):
    """Greedy decoding from left-padded inputs_embeds with an explicit KV cache."""
    lm = model.lm
    emb, mask, pos = model.embed(batch), batch["mask"], batch["pos"]
    out = lm(inputs_embeds=emb, attention_mask=mask, position_ids=pos, use_cache=True)
    past = out.past_key_values
    B = emb.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=emb.device)
    stop = torch.tensor(stop_ids, device=emb.device)
    gen = []
    nxt = out.logits[:, -1].argmax(-1)
    for _ in range(max_new_tokens):
        gen.append(torch.where(done, torch.full_like(nxt, -1), nxt))
        done |= torch.isin(nxt, stop)
        if done.all():
            break
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        pos = pos[:, -1:] + 1
        out = lm(inputs_embeds=lm.get_input_embeddings()(nxt[:, None]), attention_mask=mask,
                 position_ids=pos, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
    gen = torch.stack(gen, 1).tolist()
    res = []
    for g in gen:
        toks = [t for t in g if t != -1]
        hit = next((k for k, t in enumerate(toks) if t in stop_ids), None)
        res.append({"tokens": toks[:hit] if hit is not None else toks, "stopped": hit is not None})
    return res


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


def build_vlm(cfg, device, with_lora, seed):
    """Frozen fp16 Qwen (fp32 on CPU) + fp32 connector, optionally with fresh LoRA adapters."""
    dtype = torch.float16 if device == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device)
    lm.requires_grad_(False)
    lm.config.use_cache = False
    torch.manual_seed(seed)
    conn = Connector(pool=14 // int(round(cfg["visual_tokens"] ** 0.5))).to(device)
    model = VLM(lm, conn)
    if with_lora:
        add_lora_adapters(model, cfg)
    return model


def add_lora_adapters(model, cfg):
    lc = cfg["lora"]
    add_lora(model.lm, lc["targets"], lc["r"], lc["alpha"], lc["dropout"], seed=cfg["seed"])


def trainable_state(model):
    out = {"connector": {k: v.detach().cpu() for k, v in model.connector.state_dict().items()}}
    lora = {n: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()} for n, m in lora_layers(model.lm)}
    if lora:
        out["lora"] = lora
    return out


@torch.no_grad()
def load_trainable(model, state):
    model.connector.load_state_dict(state["connector"])
    if "lora" in state:
        layers = dict(lora_layers(model.lm))
        assert set(layers) == set(state["lora"]), "adapter layout does not match"
        for n, t in state["lora"].items():
            layers[n].lora_A.copy_(t["A"])
            layers[n].lora_B.copy_(t["B"])
