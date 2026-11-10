# UltraChat six-topic 3000-prompt data preparation

This isolated staging code prepares a new data protocol. It does not run a
model, collect fixed-prefix features, or open any sealed model result. The old
1024-prompt study remains intact.

## Frozen selection

1. Verify all three official `train_sft` Parquet shards and one `test_sft`
   shard from `HuggingFaceH4/ultrachat_200k` revision
   `8049631c405ae6576f93f445c6b8166f76f5505a` against their official LFS
   SHA-256 values. Put `dataset`, `revision`, and `parquet_sha256` filename-to-hash
   map in an `official_sha_manifest.json`. The official split sizes are
   207,865 train and 23,110 test.
2. Use the already verified first-user `train_sft.jsonl` conversion. The
   existing conversion manifest also records all four official Parquet hashes
   and may retain a `test_sft.jsonl` for provenance; the new holdout selector
   does not open that file. Assistant answers are never copied into prompts.
3. Before choosing training prompts, run `freeze_new_holdouts.py` with the
   converted `train_sft` file and its SHA-256, the prior locked `prompts.jsonl` and
   its SHA-256, the conversion manifest, and the same local Qwen3 tokenizer.
   It selects 20 new validation and 10 new sealed-test prompts **per topic**
   from `train_sft`; rejects old train/val/test exact and near duplicates; and
   writes the frozen holdout lock plus train-source IDs blocked by 5-character
   Jaccard similarity at 0.85 against all **old train/val/test prompts** and
   new holdouts. All old 1024 training source IDs and family hashes are also
   explicitly excluded. The old sealed test's first user prompt is
   decoded only from its previously locked token IDs for data hygiene and
   rechecked against its frozen family hash. New test text is used only for
   topic labeling and leakage screening. The sealed test JSONL contains token
   IDs, not responses.
4. Run `select_balanced_pilot.py` with the frozen holdout directory. It verifies
   holdout file hashes without parsing sealed test tokens, and chooses 500
   eligible training prompts per topic from `train_sft` by deterministic SHA
   rank. Ambiguous and unclassified prompts abstain. If any topic has fewer than
   500 eligible prompts, it writes a negative gate report and no training JSONL.

The six categories are the frozen `TOPICS` tuple in
[`topic_classifier.py`](../scripts/topic_classifier.py). UltraChat does **not**
provide these labels; the classifier is a research sampling instrument. Read
the exported validation and train purity audit samples
before training. If purity is inadequate, record a failed data gate and create
a new version rather than quietly relabeling or moving prompts between strata.

The old 1024 train, 64 validation, and 60 test prompt families, source IDs,
and near duplicates are excluded from the new train and new holdouts. The old
validation result is historical context,
not the selection gate for this pilot.

## Inputs and output sizes

The original 1,148-row tokenized manifest is 770,169 bytes. At the same
encoding, the new 3,180-row train/val/test manifests should be roughly 2 MB;
actual bytes and SHA-256 are recorded after preparation. Source Parquet,
first-user JSONL, and near-duplicate audit storage are additional. The
near-duplicate index retains the frozen holdout prompts in memory during data
preparation; no model output is involved.

The old `build_fixed_probe.py` and `train_fixed_probe.py` are **not** scaled
automatically by these selectors. Feature collection remains one atomic chunk
per prompt, and training currently loads every fixed-prefix example into RAM.
Measure disk and memory before launching the expanded GPU study; keep all
P0/correctness and matched-exposure gates from the new protocol.

## Tests

With PyArrow available on `PYTHONPATH`:

```powershell
$env:PYTHONPATH='C:/Users/asus/Documents/Codex/2026-09-29/n/tmp/feedback-coded-pydeps'
E:/python.exe -m unittest discover -s experiments/feedback-coded-specdec-balanced3000/tests -p 'test_*.py' -v
```

Synthetic tests cover all six categories, ambiguity, quota failure,
deterministic ranking, old source provenance, new holdout disjointness,
5-character Jaccard blocking, Parquet conversion, and exclusion of assistant
answers. They do not substitute for inspecting the actual 3000/120/60 split
and its purity audit.
