# PPLSTM — Parallel Permutation Long Short-Term Memory

**Research prototype · November 2026**

PPLSTM is a parallel-scannable, three-channel permutation-mixed recurrent memory cell being investigated as a **practical alternative to the sequential sLSTM component in xLSTM-style models**. The aim is **comparable useful language and state-tracking capability with faster training**, not necessarily smaller models or superior accuracy.

**Important:** PPLSTM changes the sLSTM recurrence; it is **not mathematically equivalent to ordinary sLSTM**. The GPU implementations below keep the *PPLSTM* recurrence, not the sLSTM recurrence.

## Three implementations — same PPLSTM mathematics

| Implementation | What changes? | Execution method | Where to look |
|---|---|---|---|
| **PPLSTM / PyTorch reference** | Original six-permutation, signed-carry cell; no accelerator-specific code | PyTorch prefix scans, eager execution | [`pplstm.py`](pplstm.py) · [proof of concept](reports/01_proof_of_concept.pdf) |
| **PPLSTM + CUDA Graphs** | **No change to the cell equations or scan code** | Captures/replays the existing PyTorch forward and backward graph, reducing launch overhead | [Graph comparison PDF](reports/03_cuda_graph_benchmarks.pdf) · [Colab notebook](benchmarks/cuda_graphs_vs_vanilla_xlstm_colab.ipynb) |
| **PPLSTM + fused T4 CUDA + Graphs** | **Same PPLSTM recurrence and trainable parameters**; replaces the memory scan, normalizer scan, output combination and their backward calculation with specialised CUDA code | Fused CUDA kernels **plus** CUDA Graph capture/replay | [10M equivalence PDF](reports/06_t4_fused_kernel_10m.pdf) · [CUDA source](kernels/t4/pplstm_fused_scan.cu) · [Kaggle notebook](benchmarks/10m_fused_equivalence_kaggle.ipynb) |

**Terminology:** “PPLSTM + Graphs” is *not* the non-accelerated original. It is the original PyTorch PPLSTM **already running with CUDA Graphs**. The fused benchmark measures the additional speedup *on top of Graphs*. The mLSTM blocks remain mLSTM blocks in both hybrids.

## Key experiment results

These are **different studies** with different setups. Compare models **within the same row/study**.

| Study | Comparison and result | Important qualification |
|---|---|---|
| [5M-token proof of concept](reports/01_proof_of_concept.pdf) | Miniature PPLSTM test PPL **54.50** vs sLSTM **51.43**; similar quality | One-layer CPU text experiment, one seed |
| [50M-token xLSTM replacement](reports/04_50m_vanilla_xlstm.pdf) | Hybrid PPLSTM **97.04** vs vanilla xLSTM **97.78** test PPL; **22,790** vs **2,187** training tok/s | Vanilla PyTorch sLSTM, **not** the official optimised sLSTM CUDA kernel; no Graphs |
| [200M-token FineWeb-Edu](reports/05_200m_fineweb_vs_transformer.pdf) | mLSTM+PPLSTM+Graphs **87.83** vs Transformer SDPA **87.60** test PPL; **35,121** vs **263,885** tok/s | 529,880 vs 528,224 parameters; one seed; context 128 |
| [10M-token fused-kernel verification](reports/06_t4_fused_kernel_10m.pdf) | PPLSTM+Graphs **34,628** vs fused PPLSTM+Graphs **111,031** training tok/s (**3.21×**); test PPL **132.52** vs **132.37** | Same hybrid architecture, data and parameter count, T4 FP32 prototype; one seed |

The [CUDA Graph speed study](reports/03_cuda_graph_benchmarks.pdf) also reports a **2.20×–6.37×** throughput advantage over *vanilla PyTorch* sLSTM+Graphs across nine tested single-block configurations. These runs do **not** establish a speed advantage over the production optimised xLSTM CUDA kernels.

## Numerical equivalence and scope

- The PyTorch PPLSTM cell's **parallel and serial computations** agree on tested outputs and gradients.
- The fused T4 implementation passed **20/20 recorded numerical/execution checks**. Same-weight full-model comparisons through training showed a largest reported logit difference of **1.24e-5** and overall relative gradient L2 difference of **5.18e-7**.
- Fusion did **not** change PPLSTM's six permutations, signed gate, gate initialization or trainable parameter structure.
- This supports **close numerical agreement and comparable training behaviour** on tested T4/FP32 workloads; it does **not** guarantee bitwise equality, identical long-run weight trajectories, or performance on every GPU.

## Documentation and reproducibility

- **[All reports and PDFs](docs/README.md)** — guide to six published reports.
- **[Implementation differences](docs/implementations.md)** — reference, Graphs, fused kernel, and what was *not* changed.
- **[Experiment evidence and limitations](docs/experiments.md)** — learning, memory tasks, speed, and comparisons.
- **[How to reproduce](docs/reproducibility.md)** — data, hardware, notebooks and verification.
- **[Runnable benchmarks](benchmarks/README.md)** — Colab/Kaggle notebooks with labels and scope.
- **[T4 fused CUDA kernel](kernels/t4/README.md)** — source, integration and supported conditions.
- **[Original standalone Python cell](pplstm.py)** / **[tests](test_pplstm.py)** / **[requirements](requirements.txt)**.

### Quick start — reference cell

```bash
pip install torch
python test_pplstm.py
python pplstm.py
```

```python
import torch
from pplstm import PPLSTM
cell = PPLSTM(width=36)
out = cell(torch.randn(2, 128, 36))
print(out.shape)  # [2, 128, 36]
```

The CUDA Graph and fused implementations are demonstrated in the notebooks and require a suitable CUDA environment. The FineWeb-Edu corpus is **not redistributed** in this repository.

## Research status

**What is supported:** the proposed PPLSTM recurrence is scan-compatible; comparable small-model language performance has been measured; the T4 fused implementation substantially improved training throughput while closely matching the PyTorch reference numerically.

**What remains open:** equivalence of *capabilities* to hidden-state-dependent sLSTM across diverse tasks; multi-seed scaling; fused-kernel correctness and robustness beyond tested GPU/dtypes/shapes; comparison against official optimised sLSTM kernels on compatible hardware. Synthetic-task results belong to the tested **combined mLSTM+PPLSTM models**, not automatically to isolated PPLSTM.

This is experimental software. The original **AGPL-3.0-only** licence applies to PPLSTM project source; other dependencies may have their own licences. See [LICENSE](LICENSE) and [commercial licensing](COMMERCIAL_LICENSING.md).

Related work: [xLSTM paper](https://arxiv.org/abs/2405.04517) · [NX-AI/xlstm](https://github.com/NX-AI/xlstm).

