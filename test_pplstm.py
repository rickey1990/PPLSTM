# SPDX-License-Identifier: AGPL-3.0-only
"""Run with: python test_pplstm.py"""
import math
import torch
from pplstm import PPLSTM, PPLSTMLanguageModel


def run():
    torch.manual_seed(21)
    cell = PPLSTM(36)
    x = torch.randn(2, 19, 36, requires_grad=True)
    a = cell(x)
    b = cell.serial_reference(x)
    error = (a - b).abs().max().item()
    assert error < 2e-5, error

    loss_a = a.square().mean()
    gradients_a = torch.autograd.grad(loss_a, list(cell.parameters()) + [x], retain_graph=True)
    loss_b = b.square().mean()
    gradients_b = torch.autograd.grad(loss_b, list(cell.parameters()) + [x])
    max_grad_diff = max((pa-pb).abs().max().item() for pa, pb in zip(gradients_a, gradients_b))
    assert max_grad_diff < 2e-5, max_grad_diff

    changed = x.detach().clone()
    changed[:, 11:] = torch.randn_like(changed[:, 11:])
    with torch.no_grad():
        causal_error = (cell(x.detach())[:, :11] - cell(changed)[:, :11]).abs().max().item()
    assert causal_error < 2e-5, causal_error

    lm = PPLSTMLanguageModel(vocab_size=512, width=36)
    tokens = torch.randint(0, 512, (2, 19))
    logits = lm(tokens)
    assert logits.shape == (2, 19, 512)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512),
                                             tokens[:, 1:].reshape(-1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lm.parameters())
    for width, length in [(3, 1), (6, 2), (12, 5), (36, 17), (96, 33)]:
        model = PPLSTM(width)
        sample = torch.randn(2, length, width)
        with torch.no_grad():
            scan = model(sample)
            serial = model.serial_reference(sample)
        assert torch.allclose(scan, serial, atol=2e-5, rtol=2e-5), (width, length)

    print(f'PASS: scan/serial outputs {error:.3g}, gradients {max_grad_diff:.3g}, '
          f'causality {causal_error:.3g}, language-model step {float(loss.detach()):.3f}; '
          '5 additional shapes passed')


if __name__ == '__main__':
    run()
