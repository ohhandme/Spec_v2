# P0 packed committed-path gate v2

Status: local implementation and CPU tests only. Real Qwen3 BF16 P0 must be rerun in a new output directory with the source SHA recorded before any P1 claim. This document interprets the frozen `program.md` §3; it does not modify that program, the frozen evaluator, prompts, oracle or models.

## Positions that can affect the output

For a packed verification of `L` proposals, let `a` be the number matching the packed Target from the beginning. Packed row `i` predicts proposal `i`; row `L` predicts the possible bonus.

| Round outcome | Hard-gated packed rows | Recorded only |
|---|---|---|
| First rejection at `a < L` | `0..a`, including the Target correction at `a` | `a+1..L` |
| All `L` accepted, bonus actually committed | `0..L`, including bonus | none |
| All `L` accepted, EOS or output budget ends the round | `0..L-1` | unused row `L` |

Each hard-gated row must have the same greedy token as an independent full-prefix Target call on precisely that row's prefix. A mismatch at a recorded-only row is retained with both token IDs, packed margin and the independent top two, but does not fail this gate. The Target output is never replaced by the independent oracle.

The complete generated token IDs, EOS and truncation signature must still equal an independent Target greedy AR request. The recorded round must commit the accepted prefix followed by exactly the packed correction or used bonus, never a rejected or hypothetical token. KV length before, raw length and cropped length must match the confirmed-prefix rule; the next round must begin at that confirmed prefix. Delayed feedback may only appear after a nonterminal verification, and has exactly eight byte indices when enabled. `full` coverage additionally requires zero/partial/full acceptance, used bonus, EOS and budget truncation. Controlled proposals exercise branches but are excluded from acceptance metrics.

`PACKED_TOKEN_GATE_VERSION = packed_committed_path_v2` is recorded in every new P0 report and failure. The optional feature gate was first introduced as `packed_valid_feature_v1`; its online-KV correction is now specified separately in [P0_PACKED_FEATURE_PROVENANCE_V2.md](P0_PACKED_FEATURE_PROVENANCE_V2.md). Neither feature gate nor a short smoke run substitutes for the full committed-path correctness gate.

## Why the archived full_v2 result is retained

The previous `gate_full_v2/failure.json` recorded prompt `six-topic-3a38acb5cff86b7f6c4af51f3af155ea5592278d57b1ebf9d365a17bc2af06fd`, raw draft, round 2: proposal 0 was token 374 while packed row 0 chose token 11. The round therefore rejected at `a=0`. Its reported position-2 disagreement (packed 27541 versus independent 69) is a post-rejection hypothetical branch and is outside the committed-path hard gate. The old source and failure report remain unchanged; this observation does **not** establish that the newly versioned full P0 passes, because later real rounds and EOS coverage still need checking.

## Code and CPU checks

`scripts/run_packed_p0.py` applies this gate in `check_rounds`, records `used_positions`, `hypothetical_positions` and the aggregate hypothetical mismatch count, and preserves full per-request evidence. `tests/test_packed.py` covers a rejected-suffix mismatch that is logged, a correction mismatch that fails, a used bonus mismatch that fails, unused budget/EOS bonus mismatches that are logged, final token/EOS/truncation mismatches that fail, and KV crop failure. The frozen Target model, BF16 dtype, independent AR oracle and input data are unchanged.

Passing CPU tests validates the gate logic only. A fresh BF16 `--coverage-mode full` run with frozen source/input/model hashes is required to determine real P0 status. The run may additionally use `--feature-provenance`; doing so does not relax any committed-path token requirement. No speed or WAN conclusion follows from this diagnostic script, which intentionally performs extra independent Target forwards.
