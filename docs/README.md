# PPLSTM report library

This is the reading order for the **six PDFs currently published in the repository**. PDFs are also kept at their original root filenames to preserve existing shared links.

| Read | Report | Covers |
|---|---|---|
| 1 | [PPLSTM proof of concept (PDF)](../reports/01_proof_of_concept.pdf) | Motivation, original recurrence, 5M-token miniature text comparison and early speed |
| 2 | [Full code appendix (PDF)](../reports/02_pplstm_code_appendix.pdf) | Standalone reference cell and mathematical scan implementation |
| 3 | [CUDA Graph speed comparisons (PDF)](../reports/03_cuda_graph_benchmarks.pdf) | Four-way eager/Graph benchmark against vanilla xLSTM; one-block setup and caveats |
| 4 | [50M-token xLSTM versus PPLSTM (PDF)](../reports/04_50m_vanilla_xlstm.pdf) | Four-block language training, 1M-interval learning table, vanilla xLSTM baseline |
| 5 | [200M-token FineWeb-Edu vs Transformer (PDF)](../reports/05_200m_fineweb_vs_transformer.pdf) | Full training curves, matched-size Transformer, synthetic memory tasks, checkpoint tables |
| 6 | [10M-token fused T4 kernel test + exact code (PDF)](../reports/06_t4_fused_kernel_10m.pdf) | Graphs versus fused+Graphs language training, same-weight numerical audits, CUDA source |

### Which PDF answers which question?

- **What is PPLSTM?** Reports 1–2, then [implementation explanation](implementations.md).
- **Did CUDA Graphs help?** Report 3.
- **Did PPLSTM replace vanilla sLSTM on text without a major quality loss?** Report 4.
- **How does the hybrid compare with a Transformer over 200M tokens?** Report 5.
- **Did the fused T4 kernel change learning or the recurrence?** Report 6, then [kernel source](../kernels/t4/pplstm_fused_scan.cu).

**Do not compare throughput across reports without checking hardware, kernel backend, graph capture, model size, token context and timed operations.** In particular, Graphs are not fused CUDA kernels.

Source notebooks: [benchmarks index](../benchmarks/README.md). Summary of limitations: [experiments](experiments.md).

