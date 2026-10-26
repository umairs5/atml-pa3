"""LoRA and OPLoRA layers.

Written by hand instead of using peft so that LoRA-16 and OPLoRA-16 get exactly
the same initial factors, and so that the projected factors are easy to export.
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # fp32 adapters on top of a (possibly fp16) frozen weight
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        self.protect_rank = 0

    def init_A(self, generator):
        # same distribution as kaiming_uniform_(a=sqrt(5)) used by peft: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        bound = 1.0 / math.sqrt(self.base.in_features)
        A = (torch.rand(self.lora_A.shape, generator=generator) * 2 - 1) * bound
        with torch.no_grad():
            self.lora_A.copy_(A)

    def set_protection(self, k):
        """OPLoRA: freeze the top-k singular bases of the starting weight."""
        W = self.base.weight.detach().float()
        U, _, Vh = torch.linalg.svd(W, full_matrices=False)
        self.register_buffer("U_k", U[:, :k].contiguous(), persistent=False)
        self.register_buffer("V_k", Vh[:k].T.contiguous(), persistent=False)
        self.protect_rank = k

    def factors(self):
        A, B = self.lora_A, self.lora_B
        if self.protect_rank:
            # keep the projection in fp32 even inside an autocast region
            with torch.autocast(device_type=A.device.type, enabled=False):
                B = B - self.U_k @ (self.U_k.T @ B)
                A = A - (A @ self.V_k) @ self.V_k.T
        return A, B

    def delta_weight(self):
        A, B = self.factors()
        return self.scaling * (B.float() @ A.float())

    def forward(self, x):
        out = self.base(x)
        A, B = self.factors()
        h = self.dropout(x).to(A.dtype)
        upd = F.linear(F.linear(h, A), B) * self.scaling
        return out + upd.to(out.dtype)


def add_lora(model, targets, r, alpha, dropout, seed, protect_rank=0):
    """Replace every nn.Linear whose name ends in one of `targets` with a LoRALinear.

    Returns the list of (name, module) pairs and the OPLoRA preprocessing time.
    """
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    gen = torch.Generator().manual_seed(seed)
    wrapped = []
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        layer = LoRALinear(getattr(parent, child), r, alpha, dropout)
        layer.init_A(gen)  # done in a fixed module order, so A is identical across runs
        setattr(parent, child, layer)
        wrapped.append((name, layer))

    svd_time = 0.0
    if protect_rank:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _, layer in wrapped:
            layer.set_protection(protect_rank)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        svd_time = time.time() - t0
    return wrapped, svd_time


def lora_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


@torch.no_grad()
def export_factors(model):
    """Effective (projected, for OPLoRA) factors, which is what inference uses."""
    out = {}
    for name, layer in lora_layers(model):
        A, B = layer.factors()
        out[name] = {"A": A.detach().float().cpu(), "B": B.detach().float().cpu(),
                     "scaling": layer.scaling}
    return out


@torch.no_grad()
def merge_factors(model, factors):
    """Add scaling * B @ A into the plain nn.Linear weights of a fresh model."""
    for name, f in factors.items():
        lin = model.get_submodule(name)
        dW = f["scaling"] * (f["B"] @ f["A"])
        lin.weight.copy_((lin.weight.float() + dW.to(lin.weight.device)).to(lin.weight.dtype))
    return model
