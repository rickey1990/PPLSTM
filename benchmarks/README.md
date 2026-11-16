# Reproducible notebooks

These notebooks are organised by the **actual implementation tested**. They were developed for free Kaggle/Colab GPUs; runs and libraries may require environment adjustments.

| Notebook | Purpose |
|---|---|
| [`200m_fineweb_transformer_kaggle.ipynb`](200m_fineweb_transformer_kaggle.ipynb) | 200M-token FineWeb-Edu: combined mLSTM+PPLSTM **with CUDA Graphs** vs parameter-matched Transformer SDPA; includes separate synthetic tasks |
| [`10m_fused_equivalence_kaggle.ipynb`](10m_fused_equivalence_kaggle.ipynb) | 10M FineWeb-Edu: Transformer vs PPLSTM **+ Graphs** vs PPLSTM **+ fused T4 kernel + Graphs**; same-weight numerical and gradient checks during training |
| [`t4_fused_short_colab.ipynb`](t4_fused_short_colab.ipynb) | T4 kernel compilation, direct numerical checks, eager and CUDA Graph timing |
| [`cuda_graphs_vs_vanilla_xlstm_colab.ipynb`](cuda_graphs_vs_vanilla_xlstm_colab.ipynb) | Four-way speed comparison with official **vanilla** sLSTM, not its production optimised kernel |
| [`10m_5000step_memory_kaggle.ipynb`](10m_5000step_memory_kaggle.ipynb) | Hybrid state tracking, delayed recall and parity with longer synthetic training |
| [`50m_memory_kaggle.ipynb`](50m_memory_kaggle.ipynb) | Hybrid 50M language and exploratory synthetic-task tests |
| [`single_block_parity_kaggle.ipynb`](single_block_parity_kaggle.ipynb) | One-block parity comparison; exploratory/inconclusive learning results |

The definitive numerical summaries are in [the PDF index](../docs/README.md). See [implementation names](../docs/implementations.md) and [reproducibility notes](../docs/reproducibility.md) before comparing speed numbers.

**Note:** Fused CUDA kernels are not the same as CUDA Graph capture. This distinction is essential when reproducing the headline 3.21× training speedup.

