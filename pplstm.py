# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 PPLSTM project author <rickeyernest1990@gmail.com>
"""PPLSTM: Parallel Permutation Long Short-Term Memory (research prototype).

A small, self-contained PyTorch implementation of the learned three-channel
permutation cell used in the prototype comparisons. No custom GPU kernels.

Input:  x [batch, time, channels] (channels divisible by three)
Output: h [batch, time, channels]

This is a new recurrent cell, not a drop-in mathematical equivalence to sLSTM.
"""
from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def shift_right(x: Tensor) -> Tensor:
    """Causal one-token shift (zero for the first position)."""
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)


def scalar_affine_scan(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Inclusive scan for x <- a*x + b, with scalar coefficients."""
    A, B = a, b
    step = 1
    while step < a.shape[1]:
        A, B = (
            torch.cat((A[:, :step], A[:, step:] * A[:, :-step]), dim=1),
            torch.cat((B[:, :step], B[:, step:] + A[:, step:] * B[:, :-step]), dim=1),
        )
        step *= 2
    return A, B


def matrix_affine_scan(A: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Inclusive scan of three-channel affine maps.

    A: [batch, time, groups, 3, 3], b: [batch, time, groups, 3].
    Returns all prefixes, where each token map acts on the *left* of the prior.
    """
    step = 1
    while step < A.shape[1]:
        next_A = torch.cat((A[:, :step], A[:, step:] @ A[:, :-step]), dim=1)
        next_b = torch.cat((
            b[:, :step],
            b[:, step:] + (A[:, step:] @ b[:, :-step].unsqueeze(-1)).squeeze(-1),
        ), dim=1)
        A, b = next_A, next_b
        step *= 2
    return A, b


class PPLSTM(nn.Module):
    """Permutation-mixed signed-memory cell with one *memory* prefix scan.

    Stabilizer and normalizer are also computed by parallel prefix operations.
    Gate scaling is shared within groups of three; channel carry is signed.
    """

    def __init__(self, width: int):
        super().__init__()
        if width < 3 or width % 3 != 0:
            raise ValueError('width must be a positive multiple of 3')
        self.width = width
        self.groups = width // 3
        self.xproj = nn.Linear(width, 4 * width)
        self.mix_logit = nn.Parameter(torch.full((4, width), math.log(3.0)))
        self.sign_proj = nn.Linear(width, width)
        self.initial_cell = nn.Parameter(torch.linspace(-1.5, 1.5, 3).repeat(self.groups))
        self.input_bias = nn.Parameter(torch.full((width,), -4.0))
        self.forget_bias = nn.Parameter(torch.full((width,), 4.0))
        nn.init.zeros_(self.sign_proj.weight)
        nn.init.constant_(self.sign_proj.bias, 2.8)

        bank = []
        for order in itertools.permutations(range(3)):
            matrix = torch.zeros(3, 3)
            for row, column in enumerate(order):
                matrix[row, column] = 1.0
            bank.append(matrix)
        self.register_buffer('permutation_bank', torch.stack(bank))

        self.selector = nn.Linear(width, self.groups * 6)
        nn.init.zeros_(self.selector.weight)
        nn.init.constant_(self.selector.bias, -1.0)
        with torch.no_grad():
            for group in range(self.groups):
                self.selector.bias[group * 6] = 3.0  # initial preference for identity

    def _token_maps(self, x: Tensor):
        """Produce the affine carry maps and stabilized normalizer gates."""
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError('Expected x with shape [batch, time, width]')
        batch, length, width = x.shape
        group_count = self.groups
        previous = shift_right(x)

        # Learned causal blend of present and previous *inputs* (not hidden state).
        coeff = self.mix_logit.sigmoid().view(1, 1, 4, width)
        gates = (self.xproj(x).reshape(batch, length, 4, width) * coeff
                 + self.xproj(previous).reshape(batch, length, 4, width) * (1 - coeff))
        input_logits, forget_logits, candidate_logits, output_logits = gates.unbind(2)
        mixed = x * self.mix_logit[0].sigmoid() + previous * (1 - self.mix_logit[0].sigmoid())

        signs = self.sign_proj(mixed).tanh().reshape(batch, length, group_count, 3)
        z = candidate_logits.tanh().reshape(batch, length, group_count, 3)

        # Shared input/forget gate values within each three-channel group.
        log_i = (input_logits + self.input_bias).reshape(batch, length, group_count, 3).mean(-1)
        log_f = F.logsigmoid((forget_logits + self.forget_bias)
                           .reshape(batch, length, group_count, 3).mean(-1))

        # Equivalent to m_t = max(m_{t-1}+log_f[t], log_i[t]), with m_0=0.
        f_prefix = log_f.cumsum(dim=1)
        m = f_prefix + torch.cummax(torch.cat((
            torch.zeros_like(f_prefix[:, :1]), log_i - f_prefix,
        ), dim=1), dim=1).values[:, 1:]
        f = (log_f + shift_right(m) - m).exp()
        i = (log_i - m).exp()

        weights = self.selector(mixed).reshape(batch, length, group_count, 6).softmax(-1)
        P = torch.einsum('btgp,pij->btgij', weights, self.permutation_bank)
        A = f[..., None, None] * P * signs[..., None, :]
        b = i[..., None] * z
        return A, b, f, i, output_logits

    def forward(self, x: Tensor) -> Tensor:
        A, b, f, i, output_logits = self._token_maps(x)
        batch, length, _ = x.shape
        prefix_A, prefix_b = matrix_affine_scan(A, b)
        c0 = self.initial_cell.tanh().reshape(1, self.groups, 3).expand(batch, -1, -1)
        c = (prefix_A @ c0[:, None, :, :, None]).squeeze(-1) + prefix_b

        # Initial normalizer is 1 within each group.
        norm_A, norm_b = scalar_affine_scan(f, i)
        n = norm_A + norm_b
        ratio = (c / n[..., None].clamp_min(1e-12)).reshape(batch, length, self.width)
        return output_logits.sigmoid() * ratio

    def serial_reference(self, x: Tensor) -> Tensor:
        """Slow reference path for checking the parallel algebra and gradients."""
        A, b, f, i, output_logits = self._token_maps(x)
        batch, length, _ = x.shape
        c = self.initial_cell.tanh().reshape(1, self.groups, 3).expand(batch, -1, -1)
        n = x.new_ones(batch, self.groups)
        states = []
        for t in range(length):
            c = (A[:, t] @ c.unsqueeze(-1)).squeeze(-1) + b[:, t]
            n = f[:, t] * n + i[:, t]
            states.append(c / n[..., None].clamp_min(1e-12))
        ratio = torch.stack(states, dim=1).reshape(batch, length, self.width)
        return output_logits.sigmoid() * ratio


class PPLSTMLanguageModel(nn.Module):
    """Minimal language-model example; not a full mLSTM + PPLSTM stack."""

    def __init__(self, vocab_size: int = 512, width: int = 36):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.norm = nn.LayerNorm(width)
        self.cell = PPLSTM(width)
        self.proj = nn.Linear(width, width)
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.embedding(token_ids) / math.sqrt(self.embedding.embedding_dim)
        x = x + self.proj(self.cell(self.norm(x)))
        return self.head(self.output_norm(x))


if __name__ == '__main__':
    torch.manual_seed(13)
    cell = PPLSTM(36)
    x = torch.randn(2, 17, 36)
    output = cell(x)
    serial = cell.serial_reference(x)
    error = (output - serial).abs().max().item()
    assert error < 1e-5, f'Scan mismatch: {error}'
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters())
    tokens = torch.randint(0, 512, (2, 17))
    logits = PPLSTMLanguageModel()(tokens)
    assert logits.shape == (2, 17, 512)
    print(f'OK | scan error={error:.3g} | output shape={tuple(logits.shape)}')