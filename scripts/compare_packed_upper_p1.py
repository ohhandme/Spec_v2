"""Compare locked P1 packed fixed-prefix rows by prompt across three seeds.

Primary contrasts are FULL-H minus C and FULL-LOGIT minus C.  The paired
bootstrap samples prompts; one-sided paired sign-flip p-values are Holm
adjusted across those two predeclared contrasts.  This is a diagnostic
acceptance analysis, not an online throughput or WAN result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feedback_codec.v2_upper_bound import UPPER_MODES
from feedback_codec.v2_p1_packed import (
    audit_evaluation_row, load_packed_p1_corpus, sha256_file,
)
from scripts.train_packed_upper_p1 import atomic_json


PRIMARY = (("full_hidden", "protocol_only"),
           ("full_logit", "protocol_only"))
SECONDARY = (("full_hidden", "edge_only"),
             ("full_logit", "edge_only"),
             ("edge_only", "raw_draft"),
             ("full_hidden", "full_hidden_zero"))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True,
                        help="parent with seed-42/<mode>/rows.jsonl etc")
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--val-corpus", type=Path, required=True)
    parser.add_argument("--val-protocol", type=Path, required=True)
    parser.add_argument("--budget-lock", type=Path, required=True)
    parser.add_argument("--cost-lock", type=Path, required=True)
    parser.add_argument("--cost-freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--permutation-repetitions", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20261101)
    return parser.parse_args()


def _read_rows(directory: Path, *, seed: int, mode: str,
               training_sha: str) -> tuple[list[dict], dict]:
    run = directory / f"seed-{seed}" / mode
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rows_path = run / "rows.jsonl"
    if (summary.get("mode") != mode or summary.get("seed") != seed
            or summary.get("training_protocol_sha256") != training_sha
            or summary.get("rows_sha256") != sha256_file(rows_path)):
        raise ValueError(f"evaluation provenance mismatch: {run}")
    rows = [json.loads(line) for line in rows_path.read_text(
        encoding="utf-8").splitlines() if line]
    if not rows or len(rows) != summary["windows"]:
        raise ValueError(f"evaluation incomplete: {run}")
    return rows, summary


def _prompt_means(rows: list[dict], *, primary: bool) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if not primary or row["primary_eligible"]:
            groups[row["prompt_id"]].append(row["accepted_count"])
    return {key: statistics.mean(values) for key, values in groups.items()}


def _contrast(values: list[float], *, bootstrap_repetitions: int,
              permutation_repetitions: int, random_seed: int) -> dict:
    if not values:
        raise ValueError("primary contrast has no eligible prompts")
    rng = random.Random(random_seed)
    observed = statistics.mean(values)
    bootstrap = sorted(statistics.mean(rng.choices(values, k=len(values)))
                       for _ in range(bootstrap_repetitions))
    if bootstrap_repetitions < 100 or permutation_repetitions < 100:
        raise ValueError("insufficient resampling repetitions")
    # The sign-flip randomization test uses one null draw for each prompt and
    # keeps all of that prompt's correlated windows/seeds together.
    exceed = 0
    for _ in range(permutation_repetitions):
        null_mean = statistics.mean(value * (1 if rng.randrange(2) else -1)
                                    for value in values)
        exceed += null_mean >= observed
    p_value = (exceed + 1) / (permutation_repetitions + 1)
    return {"prompt_count": len(values), "mean_delta": observed,
            "ci95": [bootstrap[int(.025 * (bootstrap_repetitions - 1))],
                     bootstrap[int(.975 * (bootstrap_repetitions - 1))]],
            "one_sided_sign_flip_p": p_value,
            "bootstrap_repetitions": bootstrap_repetitions,
            "permutation_repetitions": permutation_repetitions}


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError("comparison output exists; preserve the original result")
    training_sha = sha256_file(args.training_protocol)
    training = json.loads(args.training_protocol.read_text(encoding="utf-8"))
    if training.get("scope") != "P1_packed_matched_upper_training":
        raise ValueError("comparison requires locked packed P1 training")
    if (training.get("pre_val_budget_lock_sha256") != sha256_file(args.budget_lock)
            or training.get("p0_cost_lock_sha256") != sha256_file(args.cost_lock)
            or training.get("p0_cost_freeze_manifest_sha256") !=
               sha256_file(args.cost_freeze_manifest)):
        raise ValueError("pre-validation budget or P0 cost freeze changed")
    val = load_packed_p1_corpus(args.val_corpus, args.val_protocol, split="val")
    original = {example.observation.key: example for example in val.examples}
    seeds = training.get("training_seeds")
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("P1 confirmation requires at least three predeclared seeds")
    rows_by_seed_mode: dict[tuple[int, str], list[dict]] = {}
    summaries: dict[tuple[int, str], dict] = {}
    for seed in seeds:
        for mode in UPPER_MODES + ("full_hidden_zero", "full_hidden_shuffle_stratified"):
            rows, summary = _read_rows(args.evaluations, seed=seed, mode=mode,
                                       training_sha=training_sha)
            rows_by_seed_mode[seed, mode] = rows
            summaries[seed, mode] = summary
    raw_rows, raw_summary = _read_rows(args.evaluations, seed=seeds[0],
                                      mode="raw_draft", training_sha=training_sha)
    for seed in seeds:
        rows_by_seed_mode[seed, "raw_draft"] = raw_rows
        summaries[seed, "raw_draft"] = raw_summary
    for seed in seeds:
        starts = {summaries[seed, mode].get("checkpoint_initial_sha256")
                  for mode in UPPER_MODES}
        if len(starts) != 1 or None in starts:
            raise ValueError("B/C/F arms did not share identical trainable initialization")
    reference = rows_by_seed_mode[seeds[0], "protocol_only"]
    expected = {(row["prompt_id"], row["round_index"]): (
        row["target_future_ids"], row["primary_eligible"]) for row in reference}
    if len(expected) != len(reference):
        raise ValueError("duplicate validation probe key")
    val_sha = summaries[seeds[0], "protocol_only"]["val_corpus_sha256"]
    collection_sha = summaries[seeds[0], "protocol_only"]["collection_protocol_sha256"]
    if val_sha != val.corpus_sha256 or collection_sha != val.protocol_sha256 or (
            set(expected) != set(original)):
        raise ValueError("comparison val inputs differ from sealed collector")
    evaluator_sha = summaries[seeds[0], "protocol_only"]["source_sha256"]
    for key, rows in rows_by_seed_mode.items():
        mode = key[1]
        ablation_subset = mode == "full_hidden_shuffle_stratified"
        row_keys = {(row["prompt_id"], row["round_index"]) for row in rows}
        if (len(row_keys) != len(rows) or
                (not ablation_subset and row_keys != set(expected)) or
                (ablation_subset and not row_keys.issubset(expected)) or
                summaries[key]["val_corpus_sha256"] != val_sha or
                summaries[key]["collection_protocol_sha256"] != collection_sha or
                summaries[key]["source_sha256"] != evaluator_sha):
            raise ValueError(f"arms/seeds did not evaluate identical frozen probes: {key}")
        if mode in ("full_hidden_zero", "full_hidden_shuffle_stratified") and (
                summaries[key]["checkpoint_sha256"] !=
                summaries[key[0], "full_hidden"]["checkpoint_sha256"]):
            raise ValueError("F-H ablation did not reuse identical trained weights")
        for row in rows:
            row_key = (row["prompt_id"], row["round_index"])
            audit_evaluation_row(row, original[row_key],
                                 eos_token_id=val.protocol["eos_token_id"],
                                 expected_mode=mode)
            donor = row.get("donor")
            if mode == "full_hidden_shuffle_stratified":
                size = summaries[key]["ablation_audit"].get("length_bin_size")
                if type(size) is not int or size <= 0 or not isinstance(donor, dict):
                    raise ValueError("F-H shuffle lacks a locked legal donor")
                donor_key = (donor.get("prompt_id"), donor.get("round_index"))
                recipient = original[row_key].observation
                if donor_key not in original or donor_key[0] == row_key[0]:
                    raise ValueError("F-H shuffle donor is missing or same-request")
                donor_obs = original[donor_key].observation
                own_stratum = (recipient.task_type, recipient.accepted_count,
                               len(recipient.corrected_prefix) // size)
                donor_stratum = (donor_obs.task_type, donor_obs.accepted_count,
                                 len(donor_obs.corrected_prefix) // size)
                if own_stratum != donor_stratum or donor.get("stratum") != list(own_stratum):
                    raise ValueError("F-H shuffle donor leaves the task/acceptance/length stratum")
            elif donor is not None:
                raise ValueError("non-shuffle arm unexpectedly records a donor")
    full_modes = UPPER_MODES + ("raw_draft", "full_hidden_zero")
    mean_by_seed_mode = {
        (seed, mode): _prompt_means(rows_by_seed_mode[seed, mode], primary=True)
        for seed in seeds for mode in full_modes}
    all_means = {
        (seed, mode): _prompt_means(rows_by_seed_mode[seed, mode], primary=False)
        for seed in seeds for mode in full_modes}
    contrasts: dict[str, dict] = {}
    for left, right in PRIMARY + SECONDARY:
        label = f"{left}-{right}"
        prompt_ids = set(mean_by_seed_mode[seeds[0], left])
        if not prompt_ids or any(set(mean_by_seed_mode[seed, mode]) != prompt_ids
                                 for seed in seeds for mode in (left, right)):
            raise ValueError("primary prompt eligibility differs across arms/seeds")
        prompt_ids_sorted = sorted(prompt_ids)
        per_seed = {str(seed): statistics.mean(
            mean_by_seed_mode[seed, left][p] - mean_by_seed_mode[seed, right][p]
            for p in prompt_ids_sorted) for seed in seeds}
        differences = [statistics.mean(
            mean_by_seed_mode[seed, left][p] - mean_by_seed_mode[seed, right][p]
            for seed in seeds) for p in prompt_ids_sorted]
        result = _contrast(differences,
                           bootstrap_repetitions=args.bootstrap_repetitions,
                           permutation_repetitions=args.permutation_repetitions,
                           random_seed=args.random_seed + len(contrasts))
        result["per_seed_mean_delta"] = per_seed
        result["all_seed_directions_positive"] = all(value > 0 for value in per_seed.values())
        result["per_prompt_mean_delta"] = dict(zip(prompt_ids_sorted, differences))
        contrasts[label] = result
    shuffle_key = "full_hidden-full_hidden_shuffle_stratified"
    shuffle_prompt_means: dict[int, tuple[dict[str, float], dict[str, float]]] = {}
    coverage_by_seed = {}
    for seed in seeds:
        shuffled = rows_by_seed_mode[seed, "full_hidden_shuffle_stratified"]
        selected_keys = {(row["prompt_id"], row["round_index"]) for row in shuffled}
        same_f_rows = [row for row in rows_by_seed_mode[seed, "full_hidden"]
                       if (row["prompt_id"], row["round_index"]) in selected_keys]
        own = _prompt_means(same_f_rows, primary=True)
        donor = _prompt_means(shuffled, primary=True)
        if set(own) != set(donor):
            raise ValueError("F-H and shuffle have different eligible prompt subsets")
        shuffle_prompt_means[seed] = own, donor
        coverage_by_seed[str(seed)] = summaries[
            seed, "full_hidden_shuffle_stratified"]["ablation_audit"]
    prompt_ids = set(shuffle_prompt_means[seeds[0]][0])
    if prompt_ids and all(set(shuffle_prompt_means[seed][0]) == prompt_ids
                          for seed in seeds):
        sorted_ids = sorted(prompt_ids)
        seed_deltas = {str(seed): statistics.mean(
            shuffle_prompt_means[seed][0][p] - shuffle_prompt_means[seed][1][p]
            for p in sorted_ids) for seed in seeds}
        differences = [statistics.mean(
            shuffle_prompt_means[seed][0][p] - shuffle_prompt_means[seed][1][p]
            for seed in seeds) for p in sorted_ids]
        result = _contrast(differences,
                           bootstrap_repetitions=args.bootstrap_repetitions,
                           permutation_repetitions=args.permutation_repetitions,
                           random_seed=args.random_seed + len(contrasts))
        result["per_seed_mean_delta"] = seed_deltas
        result["per_prompt_mean_delta"] = dict(zip(sorted_ids, differences))
        result["coverage_by_seed"] = coverage_by_seed
        result["matched_subset_only"] = True
        contrasts[shuffle_key] = result
    else:
        contrasts[shuffle_key] = {"status": "unavailable_no_common_primary_prompt_subset",
                                  "coverage_by_seed": coverage_by_seed,
                                  "matched_subset_only": True}
    # Holm step-down adjustment for the two predeclared primary contrasts.
    ordered = sorted(PRIMARY, key=lambda pair: contrasts[
        f"{pair[0]}-{pair[1]}"]["one_sided_sign_flip_p"])
    prior = 0.0
    for rank, (left, right) in enumerate(ordered):
        result = contrasts[f"{left}-{right}"]
        prior = max(prior, min(1.0, (len(PRIMARY) - rank) *
                               result["one_sided_sign_flip_p"]))
        result["holm_adjusted_one_sided_p"] = prior
    summary = {
        "scope": "P1_packed_fixed_prefix_comparison",
        "claim": "acceptance_only_no_online_speed_or_WAN_claim",
        "training_protocol_sha256": training_sha,
        "val_corpus_sha256": val_sha,
        "collection_protocol_sha256": collection_sha,
        "seeds": seeds,
        "windows": len(reference),
        "primary_windows": sum(row["primary_eligible"] for row in reference),
        "primary_prompt_count": len(mean_by_seed_mode[seeds[0], "protocol_only"]),
        "mean_acceptance_by_seed_mode": {str(seed): {
            mode: statistics.mean(mean_by_seed_mode[seed, mode].values())
            for mode in full_modes} for seed in seeds},
        "all_window_mean_acceptance_by_seed_mode": {str(seed): {
            mode: statistics.mean(all_means[seed, mode].values())
            for mode in full_modes} for seed in seeds},
        "raw_draft_evaluated_once_shared_across_seeds": True,
        "contrasts": contrasts,
        "full_hidden_content_effect_supported": (
            contrasts["full_hidden-protocol_only"]["all_seed_directions_positive"]
            and contrasts["full_hidden-protocol_only"]["ci95"][0] > 0
            and contrasts["full_hidden-full_hidden_zero"]["ci95"][0] > 0
            and contrasts[shuffle_key].get("ci95", [float("-inf")])[0] > 0),
        "full_hidden_shuffle_coverage_by_seed": coverage_by_seed,
        "correctness": "collection_independent_AR_exact; closed_loop_P0_separate",
        "statistical_positive_signal_before_cost_gate": any(
            contrasts[f"{left}-{right}"]["all_seed_directions_positive"] and
            contrasts[f"{left}-{right}"]["ci95"][0] > 0 and
            contrasts[f"{left}-{right}"]["holm_adjusted_one_sided_p"] < .05
            for left, right in PRIMARY),
        "source_sha256": {"scripts/compare_packed_upper_p1.py":
                          sha256_file(Path(__file__))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, summary)
    print(json.dumps({"output": str(args.output),
                      "statistical_positive_signal_before_cost_gate":
                      summary["statistical_positive_signal_before_cost_gate"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
