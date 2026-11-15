# PPLSTM — Parallel Permutation Long Short-Term Memory

**Research prototype, November 2026.**

PPLSTM is an experimental way to make the **sLSTM part of an xLSTM-style model parallel across tokens**.

The original [xLSTM architecture](https://arxiv.org/abs/2405.04517) introduced two memory cells: **mLSTM**, with matrix memory that supports parallel sequence processing, and **sLSTM**, with scalar memory and recurrent hidden-state-dependent gates. An xLSTM stack can use either cell type or a combination of both. The standard sLSTM cell is harder to parallelize across time: its gates depend on the preceding hidden state, so it does not have the same straightforward input-only associative scan as mLSTM.

**The idea behind PPLSTM:** replace that sequential sLSTM recurrence with a different gated memory update that mixes groups of three memory channels using learned permutation weights. These updates compose associatively and can be evaluated by a parallel prefix scan. The goal is to retain useful sLSTM-style state tracking while improving sequence parallelism, **not** to claim that the new equations are identical to conventional sLSTM or preserve all its capabilities.

I'm sharing the code and early results for independent testing and feedback. **This is a research prototype, not a finished xLSTM replacement or a claim of state-of-the-art performance.**

## What is in this repository?

| File | Contents |
|---|---|
| `pplstm.py` | Standalone PyTorch PPLSTM cell, plus a tiny language-model example |
| `test_pplstm.py` | Scan/serial agreement, gradients, causality, training-step smoke test |
| `benchmark_gpu.ipynb` | Plain-PyTorch speed comparison against research sLSTM/mLSTM and Transformer baselines |
| `PPLSTM_Proof_of_Concept.pdf` | Short introduction, equations and experimental results |
| `PPLSTM_Code_Appendix.pdf` | Readable full code listing |
| `results/` | Numeric results from the previous training and GPU tests |
| `LICENSE` | GNU Affero General Public License version 3 |
| `COMMERCIAL_LICENSING.md` | How to request a separate proprietary licence |

## Quick start

Requires Python 3.10+ and PyTorch. CPU works; CUDA is optional.

```bash
pip install torch
python test_pplstm.py
python pplstm.py
```

Use the cell in an existing PyTorch model:

```python
import torch
from pplstm import PPLSTM

cell = PPLSTM(width=36)          # width must be divisible by 3
x = torch.randn(2, 128, 36)     # [batch, time, channels]
y = cell(x)                     # [2, 128, 36]
```

For language-model logits, see `PPLSTMLanguageModel` in `pplstm.py`.

## The idea, briefly

For each three-channel group, the model learns a soft mixture of all six possible permutations. It produces a 3×3 matrix `P_t` from the current and previous input features. With a signed carry gate `s_t`, stabilized forget/input gates `f_t, i_t`, and input candidate `z_t`, the memory update is

```
c_t = (f_t * P_t * diag(s_t)) @ c_(t-1) + i_t * z_t
```

This is `c_t = A_t c_(t-1) + b_t`. Affine maps compose associatively, so a parallel prefix scan calculates all states. The cell also retains a shared three-channel normalizer and numerical stabilizer (computed with additional prefix operations). The **underlying recurrence is changed**; this is not an exactly equivalent fast implementation of conventional sLSTM.

This implementation uses ordinary PyTorch tensor operations. There are **no custom CUDA/Triton kernels**, no CUDA Graphs and no `torch.compile`.

## What has been tested?

**Five-million-token language-model trial:** Three *small, one-layer* models were trained on the same Wikipedia-derived token split. The counts are **5M token presentations per model** (900k distinct training tokens replayed); all trained on 32-token contexts with one random seed.

| One-layer model | Validation perplexity | Test perplexity | CPU training tok/s |
|---|---:|---:|---:|
| Transformer | 56.77 | 58.90 | 86,174 |
| sLSTM cell | 53.52 | **51.43** | 53,786 |
| PPLSTM cell | **53.32** | 54.50 | 33,708 |

There is **no clear accuracy winner** between the recurrent cells. The conventional sLSTM had a better *test* result, and the PPLSTM was slower at this very short CPU training context.

**GPU speed notebook:** A separate, *untrained*, two-layer test used a shared mLSTM block followed by either sLSTM or PPLSTM, alongside a two-layer Transformer. At batch 2 × sequence 1,024, the measured times were **1,200.31 ms** (mLSTM+sLSTM), **43.59 ms** (mLSTM+PPLSTM) and **10.50 ms** (Transformer). PPLSTM was **27.54× faster than that unoptimized, Python-loop sLSTM implementation**, but the Transformer was faster than PPLSTM. GPU model details beyond `cuda:0` were not included in the results supplied for this release.

The GPU speed findings are **not** official optimized xLSTM comparisons. The 5M-text results did **not** train a combined mLSTM+PPLSTM stack. See the PDF and CSV files for all test sizes and caveats.

## What I want to test next

- Train the actual **xLSTM (mLSTM + PPLSTM)** and **xLSTM (mLSTM + sLSTM)** combinations on equal text, for several random seeds.
- Test longer contexts and models larger than the tiny prototypes used so far.
- Compare against optimized official xLSTM code, not just a Python timestep loop.
- Measure whether the extra scan work and memory are worth it on different GPUs.

Feedback, corrections and reproductions are welcome. Please share the hardware, PyTorch version and whether you changed the cell equations.

## Licence and contact

The source code is offered under **AGPL-3.0-only**. You can use, study, modify and redistribute it under those terms, including commercially. Modified covered code redistributed to others must generally remain under AGPL with corresponding source available; modified network-facing versions have additional source-access obligations under section 13. The licence is **not** a requirement to pay the author.

A **separate commercial/proprietary licence** may be available if someone needs terms different from AGPL. Contact: **rickeyernest1990@gmail.com**. This is an offer to discuss alternative terms, not a commercial licence already granted.

Please see `LICENSE` and `COMMERCIAL_LICENSING.md`. The implementation is experimental and provided without warranty. The sample Wikipedia dataset used in prior research is **not included** in this source-code release.

**Project contact:** rickeyernest1990@gmail.com

**Related work:** [xLSTM paper](https://arxiv.org/abs/2405.04517), [original xLSTM code](https://github.com/NX-AI/xlstm), and associative scans / parallel prefix computation.