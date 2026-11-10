# P1 packed fixed-prefix collection v1

This is a new, versioned collector. The historical `build_fixed_probe.py`
uses sequential full-prefix Target forwards and is **not** a valid BF16
feature source for packed P1. The packed v4 P0 report must be `passed` before
this collector starts. The collector performs its own full-request independent
AR correctness check for every train/val prompt; a new divergence stops the
run and saves a failure artifact.

## Legal observation and label boundary

For each original raw-Draft L8 packed verification round, the collector uses
`PackedSpeculativeDecoder(target, raw_draft, encoder=None, window=8,
capture_target_logits=True, feature_provenance_gate=True)`. The encoder is
absent; no feedback message affects the trajectory. The hidden vector and
top-32/logsumexp summary are taken from the **original same-forward packed
verification at its valid row**. The online KV snapshot replay and candidate
suffix intervention must both pass. A newly prefetched full-prefix Target
state is never substituted for this feature.

The raw packed result is compared with an independent greedy Target request
for the entire output, EOS and truncation included. Each probe's corrected
prefix must be a prefix of that oracle path. The next at most eight Target
tokens are copied from the oracle suffix as **offline CE labels only**. They
are not a repair input, encoder input, or `packed_trace` field. Accepted EOS
and budget-terminal rounds have no next-window probe. At most eight nonterminal
probe windows are emitted per original prompt. The `full_nonterminal_windows`
count is frozen before training; all emitted windows, including EOS in future
labels, remain in the corpus for secondary reporting.

The row's `packed_trace` retains original proposals and Target argmax IDs only
through the used decision position. Hypothetical packed rows after the first
rejection are omitted, even though v4 P0 records them diagnostically.
Counterfactual intervention outputs and numerical differences are also
omitted; only their boolean provenance gate is retained.

## Files and schema

`scripts/build_packed_probe.py` accepts only `--split train` or `--split val`
from the frozen 1000/200 tokenized pilot. The exact tokenized JSONL, metadata
and passed packed v4 P0 report SHAs are pinned in the runner. It rejects metadata indicating a
loaded sealed test. It validates the tokenized manifest SHA, tokenizer JSON,
chat templates, Target EOS, every Target/Draft model file from the v4 P0
report, and every one of the 14 v4 P0 source file SHAs. It hashes the two new
collector files in its own plan. No model download or environment mutation is
performed.

An output directory has:

- `collection_plan.json`: exact input/asset/source/parameter freeze; its file
  SHA controls resume.
- `chunks/<sha256(prompt_id)>.json`: atomic, checksum-verified per-prompt
  results. Each chunk has `request_audit`, all emitted `examples`, and both
  oracle and online provenance audit counters.
- `failures/*.json`: preserved prompt-level exception, source plan hash,
  traceback and GPU admission snapshot. Successful retry does not delete it.
- `fixed_prefix.jsonl`: merged, ordered embedded `ProbeExample.to_json()`
  records with the additional packed provenance fields below.
- `collection_protocol.json`: final scope
  `packed_fixed_prefix_collection`, version
  `packed_fixed_prefix_p1_v1`, `target_file_sha256`,
  `draft_file_sha256`, `source_sha256`, `packed_p0_source_sha256`,
  `packed_p0_report_sha256`, `prompt_manifest_sha256`,
  `prompt_manifest_meta_sha256`, `split`, `window=8`,
  `max_new_tokens=128`, `max_future_tokens=8`,
  `max_probe_windows_per_prompt=8`, `eos_token_id`,
  `collection_plan_sha256`, `fixed_prefix_sha256`, per-prompt
  `chunk_sha256`, and aggregate `audit`.

Every JSONL row adds:

```text
collection_protocol_version = "packed_fixed_prefix_p1_v1"
packed_feature_protocol_version = "packed_valid_feature_v2"
packed_feature_source = "original_packed_verify_forward"
online_feature_provenance_passed = true
independent_ar_prefix_exact = true
packed_trace = {prefix_before, proposals, used_argmax_ids,
                accepted_count, cache_length_before,
                cache_length_raw, cache_length_after, valid_prefix,
                feature_provenance}
packed_trace_sha256 = SHA256(canonical_bytes(packed_trace))
```

`canonical_bytes` is UTF-8 JSON with sorted keys, compact separators, Unicode
unescaped and NaN disallowed. `ProbeExample.to_json()` embeds the finite
Target hidden, same-forward Target logit summary, corrected prefix and offline
`target_future_ids`. The complete future may contain EOS. On resume every
chunk is rechecked against the request's independent AR output, including the
exact future suffix, provenance and chunk checksum. Final merge requires
`audit.exact_requests == audit.prompts` and
`audit.independent_ar_prefix_exact_windows == audit.probe_windows`.

## Execution boundary

Initialize the plan once, then use four mutually exclusive `--shard-index`
values with identical paths and `--shard-count 4 --resume`. The runner takes an
exclusive per-shard lock and checks three physical GPU readings before CUDA
load: no compute PID, at least 22000 MiB free, utilization below 10%. Merge
after all chunks exist with `--resume --merge-only`; this loads no model. A
stale shard lock must be checked against the actual process and released by
the operator only after evidence, never guessed from a historical PID.

The v4 P0 source list and model SHAs are immutable gates. A change requires a
new versioned gate, new plan and a new output directory. The collector is a
fixed-prefix data artifact; it does not measure packed online repair, WAN
latency, or speedup.

## Source and verification record

- Packed v4 GPU P0 report SHA256:
  `9245cd96aa5c99b7e0dd8e97c1d0a7568ba53b80c7f3ab93a5b53b82e1bec74a`
- `feedback_codec/packed_probe.py` SHA256:
  `f40afe8aed8a32a0475dc78cdd18666214111c76e53cff79a11214204d827765`
- `scripts/build_packed_probe.py` SHA256:
  `70814508affd9852a473bb10795edb6b55ff6470980735d058397d2323f69ac2`
- `tests/test_packed_probe.py` SHA256:
  `9c07547e988e5c01030c955fa1ada2b5bee17bc08ec0e3bff3898c28deb1d874`

Local syntax compilation passed. The local Python installation does not have
PyTorch, so the new CPU tests must run in the pinned remote `spec` environment
before any GPU collection. No new packed collector GPU result exists yet.
