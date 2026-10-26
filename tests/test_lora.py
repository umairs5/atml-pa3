import copy

import torch
import torch.nn as nn

from common.lora import add_lora, export_factors, lora_layers, merge_factors


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(64, 48, bias=False)
        self.down_proj = nn.Linear(48, 64, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()

    def forward(self, x):
        return self.block(x)


def make(protect_rank=0):
    torch.manual_seed(0)
    base = Tiny().requires_grad_(False)
    m = copy.deepcopy(base)
    add_lora(m, ["up_proj", "down_proj"], r=4, alpha=8, dropout=0.0, seed=1, protect_rank=protect_rank)
    return base, m


def test_same_init():
    _, lora = make()
    _, op = make(protect_rank=8)
    for (_, a), (_, b) in zip(lora_layers(lora), lora_layers(op)):
        assert torch.equal(a.lora_A, b.lora_A)
        assert torch.equal(a.lora_B, b.lora_B)


def test_projection_constraints():
    _, op = make(protect_rank=8)
    for _, layer in lora_layers(op):
        nn.init.normal_(layer.lora_B)
        dW = layer.delta_weight()
        assert dW.norm() > 0
        assert (layer.U_k.T @ dW).abs().max() < 1e-5
        assert (dW @ layer.V_k).abs().max() < 1e-5


def test_merge_and_grads():
    base, op = make(protect_rank=8)
    for _, layer in lora_layers(op):
        nn.init.normal_(layer.lora_B)
    x = torch.randn(5, 64)
    y = op(x)
    y.sum().backward()
    for _, layer in lora_layers(op):
        assert layer.lora_A.grad is not None and layer.lora_B.grad is not None
        assert layer.base.weight.grad is None

    merged = merge_factors(copy.deepcopy(base), export_factors(op))
    assert torch.allclose(merged(x), y.detach(), atol=1e-5)


if __name__ == "__main__":
    test_same_init()
    test_projection_constraints()
    test_merge_and_grads()
    print("ok")
