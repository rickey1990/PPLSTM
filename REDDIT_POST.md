# PPLSTM: a small parallel-permutation recurrent cell I've been testing

I've been working on **PPLSTM (Parallel Permutation Long Short-Term Memory)**. Original xLSTM introduced two cell types: mLSTM, whose recurrence can be parallelized, and sLSTM, whose gates depend on the previous hidden state. xLSTM models can use either or both. My aim is to replace the sequential sLSTM part with a parallel alternative, without giving up useful state tracking.

PPLSTM mixes groups of three memory channels using learned weights over permutation matrices. Each token's memory update is an affine map, which makes a parallel prefix scan possible. This is a different recurrence from sLSTM, not a drop-in mathematical equivalence. Whether it can retain the same capabilities at scale is still an open question.

I've put the equations, code and early measurements in a short PDF and GitHub-ready repository.

In a small five-million-token-presentations language-model experiment, PPLSTM was close to a sequential sLSTM baseline on held-out text. It slightly won on validation perplexity, but sLSTM did better on the separate test set, so I'm not claiming a quality win.

I also tested GPU training-step speed with plain PyTorch (no custom kernels). The prototype was much faster than an **unoptimized Python-loop sLSTM** at longer sequences — up to 27.5× at sequence 1,024 — but the small Transformer baseline was faster still. That speedup definitely isn't a claim against optimized xLSTM kernels.

This is **early proof-of-concept work**, not a production model. I'd particularly value feedback on the scan maths, how fairly the comparisons are set up, and what experiments to run next.

The code is under **AGPL-3.0-only**, with a possible separate commercial licence for projects that need different terms. Contact: rickeyernest1990@gmail.com.

[Add your repository link here]