# PPLSTM implementation guide

## The invariant: PPLSTM's proposed recurrence

Each group has three memory channels. Current and previous **input features** produce a soft mixture of six fixed permutation matrices and a signed carry:

```text
w_t = softmax(W_p u_t + b_p)
P_t = sum_j w[t,j] Pi_j
s_t = tanh(W_s u_t + b_s)
A_t = f_t P_t diag(s_t)
b_t = i_t z_t
c_t = A_t c_(t-1) + b_t
```

The exponential stabilizer, scalar normalizer and output gate are also retained. Affine maps compose associatively:

```text
(A2, b2) ∘ (A1, b1) = (A2 A1, A2 b1 + b2)
```

The equations above describe **PPLSTM**, not the standard sLSTM recurrence with hidden-dependent gates. See [`pplstm.py`](../pplstm.py) and the [code appendix PDF](../reports/02_pplstm_code_appendix.pdf) for the executable reference.

## Implementation A — PyTorch eager (reference)

- `pplstm.py` computes token maps and both memory and normalizer prefix scans with ordinary PyTorch operators.
- No CUDA Graphs, Triton or special kernels are required to run the standalone reference.
- This is the numerical comparison target and easiest implementation to inspect.

## Implementation B — PyTorch + CUDA Graphs

- **Exactly the same PPLSTM cell code and parameters as A.**
- PyTorch captures/replays forward/backward GPU operations; this reduces CPU/GPU launch overhead.
- **It does not fuse the PyTorch scan into a CUDA kernel.**
- Some notebooks use a four-block hybrid: mLSTM → PPLSTM → mLSTM → PPLSTM, with CUDA Graphs for the model's forward/backward passes.
- Source: [CUDA Graph benchmark](../benchmarks/cuda_graphs_vs_vanilla_xlstm_colab.ipynb) and [PDF](../reports/03_cuda_graph_benchmarks.pdf).

## Implementation C — T4 fused CUDA kernel + CUDA Graphs

- Retains B's CUDA Graph training path.
- **Only** replaces the PPLSTM matrix affine scan, scalar normalizer scan, state combination, output computation and associated differentiation with specialised forward/backward CUDA code.
- Leaves the six-permutation selector, signed carry gate, token projections, learned initialization, recurrence equations, mLSTM blocks and trainable parameters intact.
- Integrated through a Python adapter and compiled with CuPy NVRTC in the notebook.
- Developed and checked on Tesla T4 in FP32 at tested sequence lengths 8/128/512. No claim of portability, bit-identical output or exhaustive numerical stability.
- See [exact CUDA source](../kernels/t4/pplstm_fused_scan.cu), [Colab smoke test](../benchmarks/t4_fused_short_colab.ipynb) and [10M Kaggle equivalence trial](../benchmarks/10m_fused_equivalence_kaggle.ipynb).

### Accurate naming

| Short label | Full meaning |
|---|---|
| **PPLSTM eager** | Unfused PyTorch scans, no CUDA Graph capture |
| **PPLSTM + Graphs** | Unfused PyTorch scans, with CUDA Graph capture |
| **PPLSTM fused + Graphs** | T4 fused PPLSTM CUDA kernels, **also** with CUDA Graph capture |
| **vanilla xLSTM + Graphs** | The ordinary sLSTM PyTorch backend under CUDA Graphs; **not** the official optimised sLSTM CUDA kernel |

The headline **3.21×** gain in the 10M experiment is the last PPLSTM implementation versus **PPLSTM + Graphs**, **not** versus eager PyTorch.

## What numerical agreement does and does not mean

The fused and reference paths were evaluated at identical weights, including trained weights. Across recorded checks, same-weight model logits and gradients agreed very closely. Both separately trained runs attained nearly identical perplexity; independently optimized weights were not bitwise equal.

Numerical testing supports equivalence **in the tested conditions**; it does not prove a mathematically different sLSTM is equivalent to PPLSTM or guarantee exact floating-point results for all inputs.

Read the [10M fused-kernel report](../reports/06_t4_fused_kernel_10m.pdf) before reusing this experimental backend.

