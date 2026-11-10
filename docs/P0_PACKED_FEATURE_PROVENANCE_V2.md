# P0 packed feature provenance v2

Status: implemented and CPU tested; the real BF16 feature gate remains pending. This versioned gate supplements `packed_committed_path_v2`. It does not change the frozen `program.md`, Target model or dtype, data, independent AR oracle, acceptance rule, or full-token/EOS/KV requirements.

## Why the gate changed

The archived full-v3 feature gate compared the live valid-row feature with a new `HFPackedTargetSession` that prefills the same token prefix. After an earlier packed verification and KV crop, that new session can have numerically different BF16 KV from the live session. Its hidden and logits can therefore differ even when the current token IDs and packed committed path are correct. The archived `gate_full_v3/failure.json` contains a composite replay error; it does not by itself identify which of the replayed token, hidden, or logit comparisons failed. The original failure and the source that produced it are preserved in `autoresearch/feedback64-v2-261031/runs/p0-packed-feature-v2-20261101/source_v3_before/`.

## Hard checks in `packed_valid_feature_v2`

1. The live packed verification forward produces `PackedVerification.hidden` and `.logits`. The valid row is the prefix after the accepted proposals. Its hidden, greedy token and top-32 FP16 logits plus FP32 logsumexp/margin are taken from this **same original output object**. The decoder never substitutes a fresh-prefill replay or an AR oracle feature. An outgoing feature is required only when this round has a correction or used bonus; a terminal round with no next decision exports none.
2. Immediately before the original packed forward, P0 deep-copies the live preverification KV. The clone must have the same sequence length and no shared nonempty tensor storage pointers with the live cache, including tensors nested in cache layers. A same-query fork from this snapshot must reproduce the original valid-row hidden, full logits, and greedy token **bitwise**. If the independent copy cannot be made or reproduced, this gate fails rather than falling back to token prefill.
3. At the first rejection, a second fork from the same snapshot changes the rejected proposal while preserving the already accepted prefix and query length. The valid-row hidden, full logits, and greedy token must remain **bitwise** equal to the same-query fork. This is an intervention on an unconfirmed suffix, not a second source for the transmitted feature. If all proposals were accepted, the original query has no unconfirmed suffix after the bonus row; the suffix intervention is explicitly inapplicable, while same-query snapshot reproduction remains mandatory.
4. A fresh-prefill same-query replay is recorded as a diagnostic only. A discrepancy in its used argmax, hidden, or logit summary cannot independently fail `packed_valid_feature_v2`, because it starts from a different numerical KV history. It cannot supply the online message.

These extra forks run only under `scripts/run_packed_p0.py --feature-provenance`; they are P0 diagnostics and cannot be counted as production Target calls or latency. `check_rounds` separately enforces the original packed committed-path token, EOS, truncation, delayed-message and KV rules at every actually used verification position. Post-rejection hypothetical rows are recorded but do not fail that token gate.

## Remaining real-GPU gate

Run a new, source-hashed full BF16 P0 with `--coverage-mode full --feature-provenance`. Preserve every failure, per-request trace, used-position comparison, snapshot reproduction, suffix intervention and fresh-prefill diagnostic. Only a passed real-GPU report can validate this feature provenance version. Even a pass would not establish that FULL-H or FULL-LOGIT improves acceptance, that the packed backend accelerates generation, or that an eight-byte message is effective; those require their separately frozen experiments.
