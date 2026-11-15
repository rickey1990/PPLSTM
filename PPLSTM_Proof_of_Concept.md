# PPLSTM — Proof of Concept

**Parallel Permutation Long Short-Term Memory · Research prototype · November 2026**

PPLSTM is an experimental alternative to the sequential **sLSTM** cell used in the xLSTM family. It is intended to make sLSTM-style gated memory computations parallel across tokens without giving up useful state-tracking behaviour. **This is a research goal, not an established equivalence to sLSTM.**

## Why investigate it?

The original xLSTM introduced two cell types:

- **mLSTM:** matrix memory, with parallel sequence-processing formulations.
- **sLSTM:** scalar memory with gates that use the previous hidden state, which prevents a straightforward exact input-only parallel scan.

An xLSTM stack may contain mLSTM, sLSTM, or both. PPLSTM is a proposed replacement for the second cell type, **not** an alteration to mLSTM.

## The maths

Within each group of three memory channels, six fixed permutation matrices `Pi_1 ... Pi_6` represent all possible channel orderings. A learned token-conditioned softmax chooses their mixture:

```
w_t = softmax(W_p u_t + b_p)
P_t = sum_j w[t,j] * Pi_j
s_t = tanh(W_s u_t + b_s)
A_t = f_t * P_t * diag(s_t)
c_t = A_t @ c_(t-1) + i_t * z_t
```

The features `u_t` are derived from a learned causal combination of current and previous **inputs**. They do **not** require the previous recurrent hidden state. Within each three-channel group, the input and forget gates are shared; the initial memory is learned.

The sLSTM-inspired exponential stabilizer, normalizer, and output gate are retained:

```
m_t = max(m_(t-1) + log(f_raw,t), log(i_raw,t))
f_t = exp(log(f_raw,t) + m_(t-1) - m_t)
i_t = exp(log(i_raw,t) - m_t)
n_t = f_t * n_(t-1) + i_t
h_t = sigmoid(o_t) * (c_t / n_t)
```

The code starts with `m_0 = 0` and `n_0 = 1`. These scalar quantities are evaluated through additional prefix operations; the matrix memory transition uses **one parallel affine scan**. The whole cell is not literally one scan operation or one GPU kernel.

Let `b_t = i_t*z_t`. Two consecutive token maps combine as

```
(A2,b2) ∘ (A1,b1) = (A2 @ A1, A2 @ b1 + b2).
```

The combine rule is associative, so a parallel prefix scan finds the sequence states in logarithmic dependency depth. Rounding order can still produce small differences in floating-point arithmetic.

## Real-text experiment — 5 million tokens per model

The models were trained from scratch on matched Wikipedia-derived text, vocabulary, token order and optimizer settings. **These were single-layer models without mLSTM**, unlike the separate GPU benchmark below. 5 million counts token **presentations**, with approximately 900,000 distinct training tokens revisited across passes. One seed, 32-token training context, and 49,984 scored held-out positions per split.

| Model | Parameters | Validation perplexity ↓ | Test perplexity ↓ | CPU training tokens/sec |
|---|---:|---:|---:|---:|
| Transformer | 37,968 | 56.77 | 58.90 | 86,174 |
| sLSTM | 35,856 | 53.52 | **51.43** | 53,786 |
| PPLSTM | 34,956 | **53.32** | 54.50 | 33,708 |

PPLSTM narrowly led on validation perplexity; sLSTM was better on the separate test set. PPLSTM was slower at this short-context CPU workload. **No clear overall accuracy winner was established.**

## Separate GPU speed benchmark

The later benchmark used **untrained two-layer research prototypes**: shared mLSTM + sLSTM, shared mLSTM + PPLSTM, or two Transformer attention layers. These were timed using plain PyTorch CUDA execution, with no custom CUDA/Triton kernels or `torch.compile`. Four measured repeats per configuration.

| Batch × sequence | mLSTM + sLSTM | mLSTM + PPLSTM | Transformer | PPLSTM / sLSTM speedup |
|---|---:|---:|---:|---:|
| 32 × 32 | 47.72 ms | 22.29 ms | **7.85 ms** | 2.14× |
| 4 × 64 | 83.46 ms | 24.72 ms | **8.01 ms** | 3.38× |
| 4 × 128 | 154.85 ms | 26.87 ms | **7.98 ms** | 5.76× |
| 4 × 256 | 300.61 ms | 29.56 ms | **8.08 ms** | 10.17× |
| 4 × 512 | 607.79 ms | 39.10 ms | **7.60 ms** | 15.54× |
| 2 × 1,024 | 1,200.31 ms | 43.59 ms | **10.50 ms** | 27.54× |

The PPLSTM prototype becomes much faster than **this specific unoptimized Python-timestep sLSTM baseline** at longer sequences; the Transformer remained fastest in all tested configurations. These are *not* measurements against official optimized xLSTM kernels, and the GPU hardware model was not supplied in the raw result table.

## What is proved, and what is not?

Verified locally: the parallel and serial calculations of the **PPLSTM recurrence** agree, including gradient comparisons and causal-output checks. This establishes implementation consistency, **not** equivalence to ordinary sLSTM.

Still to test: equal-token-budget training of full mLSTM+sLSTM and mLSTM+PPLSTM stacks; multiple seeds; larger models; longer training contexts; comparisons with optimized xLSTM on comparable GPUs; and memory/throughput scaling.

**PPLSTM is an early prototype made available for independent testing and refinement. It is not a claim of a proven faster or better full-scale xLSTM model.**

**Contact:** rickeyernest1990@gmail.com  
**Licence:** AGPL-3.0-only; separate commercial terms may be discussed.
