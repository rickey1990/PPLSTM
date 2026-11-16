# Specialised PPLSTM T4 fused scan

Source: **[`pplstm_fused_scan.cu`](pplstm_fused_scan.cu)**. This is the forward/backward CUDA source used in the tested 10M-token experiment.

The original PPLSTM parameterisations and recurrence are preserved. This fused code receives token-map matrices, memory writes, stabilized scalar gates, output gates and initial memory from the original PyTorch gate-generation logic; it combines the memory-state scan, normalizer and output computation in fewer GPU operations and defines a matching backward calculation.

**What it does not replace:** six-permutation selector, signed gate, QKV mLSTM blocks, optimizer, or the PPLSTM recurrence itself.

### Test status

- Developed and tested on **NVIDIA Tesla T4** with FP32 calculations.
- Direct tests at lengths **8, 128, 512** passed the configured output/gradient checks.
- Same-weight **full-model trained-weight** audits passed at the start and through a 10M-token trial.
- Real training measured **34,628 tok/s (PyTorch + Graphs)** vs **111,031 tok/s (fused + Graphs)**, a **3.21×** speedup.
- A parameter-matched Transformer SDPA in the same study measured **263,108 tok/s**.
- Precision agreement is very close **but not bitwise identical**; this is not yet a general-purpose, production-hardened kernel.

**How to run:** use the [short Colab diagnostic](../../benchmarks/t4_fused_short_colab.ipynb) or [10M Kaggle experiment](../../benchmarks/10m_fused_equivalence_kaggle.ipynb) for the Python CuPy NVRTC compile and PyTorch autograd adapter. The raw CUDA file alone is **not** a standalone pip-installable extension.

Full results and embedded Python adapter: [10M report PDF](../../reports/06_t4_fused_kernel_10m.pdf).

Project source is AGPL-3.0-only unless otherwise noted; check individual dependency licences.

