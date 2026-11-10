# B/C/F-H Transformer repair candidate (local code)

This candidate keeps the repaired v13 B/C/F-H information boundaries and
replaces the output residual MLP with a two-layer, width-128 causal Transformer.
It does **not** change Target labels, the verifier, top-k semantics, or the
correction boundary. The Draft backbone and vocabulary head remain frozen.
F-H sees the same-boundary valid Target hidden and is a diagnostic upper bound;
it is not an 8-byte feedback arm.

## Files

- `feedback_codec/v10_transformer_bcfh.py`: matched three-arm module. A
  zero-initialized output projection starts every arm at raw Draft logits.
- `scripts/load_transformer_v13_windows.py`: index and read only the exact
  repaired train/validation manifests. It checks the frozen parent prompt
  roster, artifact hashes, legal boundary hidden references, and 7982/1598
  selected windows. `verify_train_receipt` binds the 1000-member index to a
  completed full-collection audit receipt.
- `scripts/transformer_bcfh_experiment.py`: equal-exposure training and
  free-running paired next-window validation. It does not dispatch work or
  collect data.
- `scripts/transformer_bcfh_closed_loop.py`: full-request semantic evaluation
  against the **actual recorded original packed Vanilla SD** requests. It
  first requires exact raw replay and exact Target AR output for every arm.
- `tests/test_v10_transformer_bcfh.py`: causal mask, legal information,
  matched initialization, gradient and inference-path contracts.

## Frozen protocol

1. Use the **repaired** 1000 train requests (7982 legal windows) and 200
   validation requests (1598 legal windows). The old 984/199 subset cannot
   pass `audit_windows`. Keep test sealed.
2. Before training, validate the complete train receipt with
   `verify_train_receipt`. Supply its SHA-256 and the separately verified
   frozen Draft checkpoint SHA-256 to `train_three_arms`. Confirm the complete
   Draft asset inventory against the original model lock before calling it;
   the runner additionally pins a small runtime parameter fingerprint.
3. Train B, C, F-H independently from matched seed-42 initialization, five
   epochs each, 39,910 updates per arm. Use the original first-position
   weighted CE and local straight-through code selection. No MLP checkpoint
   is loaded. The isolated output directory records source hashes, an
   immutable lock, resumable optimizer state and final checkpoints.
4. Only after all three training terminals exist, read the 200-prompt
   validation set and run `evaluate_three_arms`. Training and greedy
   inference both use the full **current** corrected-window Draft history
   through the same causal Transformer. No future Target token enters the
   forward path.
5. For the next-window diagnostic, `load_original_raw_next` joins every
   selected window to the next **actually collected packed Vanilla SD** round.
   The evaluator checks a fresh frozen-Draft replay against those recorded
   proposals. Compare arms on identical corrected prefixes and full eight-token
   eligible windows, with prompt-equal means and a paired family bootstrap.
6. The **formal goal** is full-request accepted Draft tokens per verification
   round versus original speculative inference on the same 200 requests.
   `load_original_raw_requests` reads all original rounds and independent
   Target AR outputs; the independent 200-request full-field validation
   acceptance receipt must bind all request manifests. `evaluate_closed_loop`
   first reproduces every original
   raw proposal, acceptance count and output exactly, then runs B/C/F-H from
   the original prompt with feedback starting after the first verification.
   It requires each arm to preserve the exact Target AR output. An arm passes
   only when its paired point gain is positive and the 95% interval lower
   bound is above zero. A positive next-window diagnostic alone is not a pass.

`ExperimentLock` defaults to depth 2 / width 128 / 4 attention heads. The
implementation also permits bounded 1-layer/64-wide alternatives, but these
are new experiments with separate locks and outputs. The code cannot promise
an acceptance gain before the repaired data, training, and paired validation
actually complete. The closed-loop runner recomputes full prefixes and Target
verification sequentially; its acceptance semantics are checked against the
original packed trace, but it is not a packed repair throughput, speed or WAN
result.

## Integration boundary

The local library expects a frozen Draft backend with `teacher_hidden` and
`model.get_{input,output}_embeddings()`, `CodecConfig` with the locked Draft
and Target dimensions, and `PilotWindow` sequences from the index reader.
The caller must perform the fresh two-scan GPU UUID / no-compute-PID /
free-memory / utilization admission and cross-root `flock` **before** invoking
training. Do not start this candidate through an old task28/task34 queue or
reuse their output directories. The current code only prepares this isolated
candidate; it does not launch any job.
