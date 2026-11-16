# Experiment record and interpretation

This page is an **index to measured reports**, not a claim that experiments with different training data, models or optimisations are directly comparable.

## Text learning

| Experiment | Models | Reported result | Restrictions |
|---|---|---|---|
| 5M tokens, one-layer miniature CPU | sLSTM and PPLSTM, plus Transformer | sLSTM test PPL 51.43; PPLSTM 54.50 | One seed, context 32, original research-style cells |
| 50M tokens, four blocks | Vanilla xLSTM: 2 mLSTM + 2 sLSTM; alternative: 2 mLSTM + 2 PPLSTM | Test PPL **97.78 vs 97.04**; throughput **2,187 vs 22,790 tok/s** | Baseline uses vanilla PyTorch sLSTM, no CUDA Graphs or optimised sLSTM kernel; PPLSTM has more parameters |
| 200M tokens, FineWeb-Edu, context 128 | PPLSTM+Graphs hybrid (529,880 params) vs Transformer SDPA (528,224 params) | Test PPL **87.83 vs 87.60**; throughput **35,121 vs 263,885 tok/s** | One seed; the Transformer is much faster at this small model/context |
| 10M tokens, T4 fused verification | PPLSTM+Graphs vs fused PPLSTM+Graphs vs Transformer SDPA | Hybrid test PPL **132.52 vs 132.37**; throughput **34,628 vs 111,031 tok/s** (**3.21×**) | Same PPLSTM architecture/parameter count; one seed; different training corpus/split from 200M experiment |

The **5M study**: [PDF](../reports/01_proof_of_concept.pdf) · [CSV](../results/real_text_5m.csv).  
The **50M study**: [PDF with 50 1M-interval checkpoints](../reports/04_50m_vanilla_xlstm.pdf).  
The **200M study**: [full PDF](../reports/05_200m_fineweb_vs_transformer.pdf).  
The **10M fused study**: [full PDF](../reports/06_t4_fused_kernel_10m.pdf).

## Performance measurements

- The original plain-PyTorch GPU benchmark found large advantages against an **unoptimised sequential sLSTM timestep loop**. See [CSV](../results/gpu_speed_user_reported.csv).
- The later four-way CUDA Graph sweep measured PPLSTM+Graphs **2.20×–6.37× faster** than vanilla xLSTM+Graphs across nine configurations. [Report](../reports/03_cuda_graph_benchmarks.pdf). This is **not** an official optimised xLSTM kernel comparison.
- The T4 fused scan showed a **3.21× improvement in real training throughput** over the **already-graphed** PPLSTM hybrid in the 10M study, with numerically close outputs and gradients. [Report](../reports/06_t4_fused_kernel_10m.pdf).

## Synthetic memory and reasoning

The 200M report also includes *separately trained* synthetic-task models; they are **not zero-shot tests of the 200M language checkpoints**. At 512-token evaluation after 128-token training, selected results were:

| Task | mLSTM + PPLSTM hybrid | Transformer |
|---|---:|---:|
| State tracking mod 4 | **89.06%** | 27.93% |
| Delayed recall | **100%** | 82.23% |
| Key–value recall | **100%** | 22.07% |
| Key–value updates | **93.55%** | 19.73% |
| Parity | ~49% | ~49% |

The two-hop and three-hop generators admit a shortcut around **25% accuracy**; their near-chance results are **not** evidence of learned multi-hop reasoning. These tests cannot independently attribute success to PPLSTM rather than mLSTM or their combination. See [complete PDF including chance analysis](../reports/05_200m_fineweb_vs_transformer.pdf).

Additional Kaggle memory trials are available in [benchmarks](../benchmarks/README.md). The longer 5,000-step hybrid task tests reported ~96% mod-4 state tracking and 100% delayed recall at 512 tokens, but parity remained near chance. The single-block parity notebook was exploratory and does not establish a general architecture limitation.

## Numerical results: the fused kernel

The T4 trial reported **20/20 checks passed** including direct and same-weight comparisons during training:

| Same-weight check | Maximum reported |
|---|---:|
| PPLSTM output max absolute difference | 8.94e-8 |
| Full-model logit max absolute difference | 1.24e-5 |
| Full-model loss max absolute difference | 4.77e-7 |
| Overall gradient relative L2 difference | 5.18e-7 |

These are close but **not bitwise identical**. Tested conditions do not guarantee numerical stability across all shapes, precisions, inputs or training budgets.

## Open research questions

1. How much of conventional sLSTM's hidden-dependent control can PPLSTM approximate on controlled, multi-seed state tasks?
2. What is the **isolated** contribution of PPLSTM compared with mLSTM-only and mLSTM+sLSTM?
3. How does this fused kernel behave on longer contexts, different trained states and newer GPUs?
4. How does it compare to **optimised official xLSTM kernels** on matching hardware?

**Research success criterion:** comparable useful capability and faster execution; PPLSTM is **not required to outperform** sLSTM or the Transformer on every task.

