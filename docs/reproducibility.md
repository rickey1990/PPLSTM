# Reproducing the experiments

Start with the [benchmarks index](../benchmarks/README.md). These are runnable experiment notebooks, not a single turnkey command that reproduces every published number on every machine.

## Requirements and data

- **Reference cell:** PyTorch on CPU or CUDA. Run `python test_pplstm.py`.
- **CUDA Graph comparison:** GPU supporting the notebook's CUDA Graph path. The reference official sLSTM comparison uses the vanilla backend because its production CUDA kernel is not available on a Tesla T4 (compute capability 7.5).
- **T4 fused kernel:** NVIDIA Tesla T4; CuPy/NVRTC compilation; tested FP32. The source is at [`kernels/t4/pplstm_fused_scan.cu`](../kernels/t4/pplstm_fused_scan.cu). Other hardware is not yet validated.
- **FineWeb-Edu training:** the 10M and 200M notebooks expect the FineWeb-Edu parquet dataset attached in Kaggle (previous experiments called it `cartman`). Dataset is not committed here.
- **Token budget:** “10M”, “50M”, “200M” denote token **presentations per model**. The 200M study reuses training tokens rather than representing 200M distinct source tokens.
- **Checkpoints:** PDFs document scores but do not contain trained weight files. Complete previous experiment ZIP archives were not all committed to GitHub.

## Which notebook to run?

| Goal | Notebook |
|---|---|
| Baseline 200M PPLSTM+Graphs vs Transformer | [200M FineWeb-Edu](../benchmarks/200m_fineweb_transformer_kaggle.ipynb) |
| Compare original-graph vs fused-graph, with trained-weight audits | [10M fused equivalence](../benchmarks/10m_fused_equivalence_kaggle.ipynb) |
| Small T4 kernel numerical and performance smoke test | [T4 fused Colab](../benchmarks/t4_fused_short_colab.ipynb) |
| Compare single-block vanilla xLSTM and PPLSTM with CUDA Graphs | [Graph comparison](../benchmarks/cuda_graphs_vs_vanilla_xlstm_colab.ipynb) |
| Hybrid synthetic state/recall memory experiments | [10M + 5k steps](../benchmarks/10m_5000step_memory_kaggle.ipynb) · [50M memory](../benchmarks/50m_memory_kaggle.ipynb) |
| Exploratory single-block parity training | [Parity notebook](../benchmarks/single_block_parity_kaggle.ipynb) |

## Tests to keep when changing implementation

1. Verify **serial vs parallel PPLSTM** outputs, input gradients and parameter gradients.
2. Compare **PPLSTM+Graphs vs fused+Graphs with identical weights**, on both randomly initialized and trained states.
3. Verify finite full-model backward gradients and future-token causality.
4. Run identical training batches and optimizer settings with matched parameter counts, reporting loss, validation/test perplexity and throughput.
5. Record CUDA device, precision, sequence length, batch, timed operations, Graph use and kernel backend.
6. Treat any failed or skipped numerical test as **not validated**; don't substitute untested kernel results.

## Documentation boundaries

- The [reference cell](../pplstm.py) is independently runnable and AGPL-3.0-only.
- Fused kernel source and notebook adapters are research code and may rely on separately licensed packages. Verify upstream dependency terms independently.
- The report PDFs are experiment records, not benchmarks of the latest optimised official xLSTM release.

