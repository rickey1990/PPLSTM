# Experiment notes and provenance

These are records of earlier prototype experiments, not claims of optimized performance. The aim is to test whether the hidden-state-dependent sLSTM portion of an xLSTM-style stack can be replaced by a parallel-scannable permutation-mixing cell without sacrificing useful state-tracking ability. The experiment does not yet prove equal functionality, and standard xLSTM stacks need not contain both cell types.

## Real-text results

Source file: `results/real_text_5m.csv`, copied from the completed local five-million-token comparison.

- Training was carried out on CPU with a one-layer model per architecture (Transformer, sLSTM, PPLSTM).
- One training seed, 4,884 steps, 5,000,000 presented tokens per model; 32-token context, up to 32 samples per batch.
- The training text used about 900,000 distinct tokens from an earlier, local Wikipedia-derived corpus, repeated across training passes. The original corpus is not redistributed in this project.
- Held-out validation and test each scored 49,984 positions using a 512-token SentencePiece vocabulary.
- The PPLSTM cell is **not** a mathematically identical acceleration of sLSTM, and the Transformer is architecturally different.
- The source model used in the code release was independently compared on identical parameters/input to the previous GPU notebook's PPLSTM cell; maximum output difference was **zero** for that test.

## GPU speed results

Source file: `results/gpu_speed_user_reported.csv`, transcribed from the author's supplied benchmark output.

- The speed notebook used *untrained*, two-layer research implementations with a shared mLSTM layer and then sLSTM or PPLSTM, plus two causal attention layers for the Transformer.
- PyTorch CUDA device identifier in the supplied output: `cuda:0`; specific GPU model not recorded in that table.
- Times: median of 4 measurements, following 2 warm-up iterations. Each timed step included forward, cross-entropy loss, backward and unfused AdamW update. Model creation and random input generation were excluded.
- No custom Triton/CUDA kernels, CUDA Graphs, compile or explicit FlashAttention. Standard PyTorch CUDA math libraries still do the underlying computation.
- The 27.54x advantage at batch 2, sequence 1,024 is **only versus the notebook's unoptimized Python-timestep sequential sLSTM cell**. The Transformer was faster than PPLSTM.
- Because the language-quality results are from one-layer cell models and speed results are from mixed two-layer models, their outcomes should not be combined to claim a quality/speed win for a single trained architecture.

## Verification performed for this code release

`python test_pplstm.py` passed the scan vs serial test, gradient agreement, causality test, and a small language-model backward pass on CPU. A separate comparison against the earlier notebook implementation using the same weights and input gave 0.0 maximum output difference. These checks validate the implementation of the proposed recurrence; they do not validate natural-language ability or optimized GPU performance.