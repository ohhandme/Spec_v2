# Balanced-3000 runtime plan (implementation audit)

This plan applies only to the isolated `feedback-coded-specdec-balanced3000` experiment. It does not launch a GPU job. The final topic taxonomy and dataset selection are separate gates; no collection starts until those gates pass and their hashes are frozen.

## Manifest and sealed test

The first remote prompt manifest is `train_val_prompts.jsonl` and contains **train (3000) and new validation (120) only**. Its metadata must set `prompts_jsonl_sha256` and, if present, `output_sha256` to that exact train+val file; counts must report train=3000, val=120, test=0. Keep the new test prompt file and its SHA lock on the local machine. `run_real_p0.py` checks the manifest hash and train count; `run_fixed_probe_gpu_p0.py` checks the same hash; `build_fixed_probe.py` validates the manifest and selects the requested split. None requires test rows for the first train/val run.

**Current one-seed pilot stops after validation. Do not unseal, upload, collect, or evaluate the test split in this run.** If a later preregistered multi-seed study passes its validation gate, assemble a *new* append-only `train_val_test_prompts.jsonl` manifest with the previously sealed test rows, record both source hashes, and rerun real P0 once against its changed manifest SHA. Then collect test with `--split test --allow-test`. `evaluate_fixed_probe.py` accepts the original frozen training protocol and checkpoints with that new test collection: it checks matching Target/Draft model hashes, the unchanged checkpoint/training-protocol SHA, train exposure/init/steps, and that evaluation corpus SHA differs from train corpus. It does not require the evaluation manifest SHA to equal the train manifest SHA. Do not edit its guards.

The local `scripts/unseal_final_test_manifest.py` requires an existing `balanced3000_validation_selection_v1` lock whose `state` is `selected_for_one_time_test`, the active/test hashes match, and all three report hashes and gate booleans are present. The person or script producing that lock must first independently audit actual 42/43/44 validation reports, checkpoint fairness, and full correctness. The unseal helper checks provenance and split disjointness; it does **not** calculate D-C/D-E from reports. A negative gate leaves the test file unopened. Run the helper only once, with the frozen local sealed-test SHA:

```text
E:/python.exe scripts/unseal_final_test_manifest.py \
  --active-prompts <local-data>/train_val_prompts.jsonl \
  --active-meta <local-data>/manifest.json \
  --sealed-test <local-holdouts>/new_test_sealed.jsonl \
  --sealed-test-sha256 <frozen-sha256> \
  --validation-selection-lock <local-results>/validation_selection_lock.json \
  --output-dir <local-results>/final-test-unsealed
```

Verify the emitted SHA against the local file, transfer **only then** the two output files to the isolated remote project, and verify remote `sha256sum` matches. On an admitted idle GPU, run the existing real P0 against the *new* manifest (a new output directory is required):

```text
/mnt/files2/dream/sd-lora-preexp/envs/spec/bin/python scripts/run_real_p0.py \
  --target /mnt/files2/dream/feedback-coded-specdec/models/Qwen3-8B \
  --draft /mnt/files2/dream/specbranch-reproduction/models/Qwen3-0.6B \
  --target-download-status /mnt/files2/dream/feedback-coded-specdec/models/qwen3-8b-download-status.json \
  --dataset <remote-final>/train_val_test_prompts.jsonl \
  --dataset-manifest <remote-final>/manifest.json \
  --output <remote-run>/real-p0-test --device cuda:<admitted-index> \
  --request-count 4 --max-new-tokens 24
```

Require `real-p0-test/report.json` state=passed, BF16, new manifest SHA, model/source SHA and exact Target AR. Next, with a fresh idle admission, collect and atomically verify all 60 test prompts:

```text
/mnt/files2/dream/sd-lora-preexp/envs/spec/bin/python scripts/build_fixed_probe.py \
  --target /mnt/files2/dream/feedback-coded-specdec/models/Qwen3-8B \
  --draft /mnt/files2/dream/specbranch-reproduction/models/Qwen3-0.6B \
  --dataset <remote-final>/train_val_test_prompts.jsonl \
  --dataset-meta <remote-final>/manifest.json \
  --p0-report <remote-run>/real-p0-test/report.json \
  --split test --allow-test --thinking-mode disabled \
  --output <remote-run>/fixed-probe-test --device cuda:<admitted-index> \
  --window 8 --max-new-tokens 32
```

For each selected seed, use the original **train** `training_protocol.json` and its three mode-matched `trainable.pt` checkpoints, with the new **test** collection files. Run in distinct output folders; use the same bootstrap seed/replications and identical test corpus for every seed:

```text
/mnt/files2/dream/sd-lora-preexp/envs/spec/bin/python scripts/evaluate_fixed_probe.py \
  --target /mnt/files2/dream/feedback-coded-specdec/models/Qwen3-8B \
  --draft /mnt/files2/dream/specbranch-reproduction/models/Qwen3-0.6B \
  --corpus <remote-run>/fixed-probe-test/fixed_prefix.jsonl \
  --collection-protocol <remote-run>/fixed-probe-test/collection_protocol.json \
  --training-protocol <remote-run>/train/training_protocol.json \
  --edge-checkpoint <remote-run>/train/seed-<seed>/edge_only/trainable.pt \
  --protocol-checkpoint <remote-run>/train/seed-<seed>/protocol_only/trainable.pt \
  --target-checkpoint <remote-run>/train/seed-<seed>/target_8B/trainable.pt \
  --split test --allow-test --output <remote-run>/test-eval/seed-<seed> \
  --device cuda:<admitted-index> --bootstrap-repetitions 2000 --seed 20261031
```

Before test collection, validate the full manifest still has train=3000, val=120, test=60, six topic-balanced test strata, no family/token/ID crossover, and exact sealed-test SHA. The test collector's own `--allow-test` and evaluator's `--allow-test` are independent explicit gates. No test-driven retry or candidate change follows this one-time evaluation.

## GPU admission

Before each worker, take three separate `nvidia-smi` samples. A card is eligible only when all three have zero compute PIDs, at least 22000 MiB free and utilization under 10%. Recheck immediately before process launch. Admission does not reserve a card against another user; the controller watches for a foreign compute PID on any card it occupies, then signals only the exact self-owned worker process group and preserves completed chunks. Other users' processes are never stopped. If fewer than four cards qualify, run fewer shards concurrently and admit the remaining shards when cards become free.

## Gates and collection

1. Verify model-file hashes, prompt manifest SHA, source hashes, and no existing live experiment job. Run `run_real_p0.py` on the new train+val manifest; require CUDA BF16 `state=passed` and exact AR. Run `run_fixed_probe_gpu_p0.py` with the matching P0 report; require CUDA `state=passed` for B/C/D correctness and gradients.
2. Run `build_fixed_probe.py --initialize-only --shard-count 4` exactly once for train. This writes `collection_plan.json`, including model/source/data/P0 SHA and selected prompt digests. Freeze its SHA. Do a small smoke on the same plan with `--resume --max-prompts-this-call` on each admitted shard; then resume normally. Each prompt's shard is `sha256(prompt.id) % 4`; one worker per shard and its `shard-N.lock` prevent duplicate owners. A prompt chunk is written atomically and checked against prompt/plan/shard SHA.
3. Run the four shard workers concurrently on distinct eligible GPUs. A restarted worker uses `--resume` with the original plan. Do not clear a shard lock without checking its PID, command, plan SHA and process exit. A child failure other than documented card contention stops the queue for diagnosis; it does not trigger an unexamined retry.
4. After all 3000 atomic chunks exist and no shard worker is live, run `--merge-only --resume`. This verifies every chunk checksum, all prompt IDs, identical EOS, and writes `fixed_prefix.jsonl` and `collection_protocol.json` atomically. Verify 3000 prompt coverage and final SHA. Repeat for the 120 validation prompts, with a distinct output directory and plan SHA. Never merge partial shards.

The collector is already sharded and restartable. It loads Target and Draft separately on each GPU; four workers need approximately four copies in host RAM and up to four concurrent model-file hash passes. The prior server had about 239 GiB available RAM and 164 GiB free disk at the last read, but check both again at launch. Measure chunk bytes and peak RSS after the first 100-300 prompts; stop before exhaustion. Four-card collection has an ideal time of roughly 1.3-2 hours for 3000 prompts based on the old ~1h49/1024-prompt run, plus model loading and merging. This is an estimate, not a measured new throughput.

## Current one-seed training and validation

After new train collection, lock its actual window count as one no-replacement epoch, LR=1e-4, weight decay=0.01, CE-only and seed 42 **before viewing new validation acceptance**. The existing trainer now supports an explicit `--single-seed-pilot`; `--seeds 42` without that flag is rejected so the output cannot masquerade as the preregistered three-seed P1 result. Run B/C/D sequentially on one admitted card with `--seeds 42 --single-seed-pilot`, `--steps <actual-train-window-count>`, `--prefix-weight 0`, `--temperature 1`, and a passed matching `--p1-gpu-p0-report`. The unchanged loop checks identical edge initialization and prompt-window exposure across B/C/D; atomic `last.pt` supports matching-protocol resume. Use the ordinary final `training_summary.json` and checkpoint SHA evidence. Evaluate seed 42 once on the frozen new validation corpus using the existing `evaluate_fixed_probe.py`, inspect exact-token/EOS/truncation correctness and paired D-C/D-E. Report this as a **single-seed mechanism pilot**, with no sealed test or stable multi-seed claim.

At the old ~0.124 s training step per arm, ~27k train windows imply ~55 minutes per B/C/D arm or about 2.75 GPU hours for one seed. Four-way train collection might take ~1.3-2 hours, validation collection/eval plus P0/model loads and merging another ~0.5-1 hour. A tentative **4-5 hour** wall-clock after data freeze is plausible if four cards remain idle, but measure the new per-prompt and per-step rates before giving a finish time.

## Optional later multi-seed study (not authorized in this pilot)

Before looking at validation acceptance in a later three-seed study, lock the train-window count, LR, weight decay, CE-only, seeds 42/43/44, and all hashes. The stock `train_fixed_probe.py` runs all three seeds and B/C/D modes sequentially under one `training_protocol.json`. Splitting its CLI into separate GPU processes requires an independently tested coordinator preserving one shared protocol and per-seed/arm ownership. Existing atomic `last.pt` permits verified `--resume`; completed arms have SHA-checked summaries. Compare exposure and initial-edge SHA across B/C/D for each seed.

For each seed, `evaluate_fixed_probe.py` loads three matched checkpoints and one validation corpus, so it is naturally parallelizable **across seeds** on up to three eligible cards, with a distinct output directory per seed. It refuses an existing output; inspect any partial result before a retry. Use the same frozen validation corpus, bootstrap repetitions and paired prompt grouping for all seeds. Report D-C and D-E with topic-stratified/macro estimates; the existing evaluator supplies E and paired comparisons. No WAN or speedup conclusion follows from this sequential reference backend.

The later three-seed study would need roughly 7-9 GPU hours for nine ~27k-step arms under the stock trainer, in addition to collection and validation. That estimate is not the current one-seed pilot.

## Recovery and evidence

Record actual GPU UUID, PID, process start tick, command, source/data/model SHA, plan SHA, prompt counts, log path, exit reason and retry in the new autoresearch ledger. An interrupted shard retains valid chunks; restart from the same plan and source only. An OOM or token mismatch requires preserving the failure and identifying the cause before a code change; changed source must rerun affected P0 and collection gates with an explicitly new plan. The previous 1024 study, its test, and other users' jobs remain untouched. No GitHub push or publication.
