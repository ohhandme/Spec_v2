"""Predeclared v11 prompt/family-clustered fixed-prefix statistics.

Window rows are not independent observations. Each prompt contributes its
mean over eligible windows, and whole families are resampled within source.
The seed-level intervals condition on the three fixed training seeds.
"""

from __future__ import annotations

from collections import defaultdict
import random
from statistics import mean
from typing import Iterable, Mapping


TRAIN_SEEDS = (42, 43, 44)
ARMS = ("raw_draft", "B", "C", "F-H", "F-CURRENT-TOP32")
PRIMARY_COMPARISONS = (("F-H", "C"), ("F-H", "B"))
SECONDARY_COMPARISONS = (("F-CURRENT-TOP32", "C"),
                         ("F-CURRENT-TOP32", "B"))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 11022026


def prefix_acceptance(proposed: Iterable[int], future: Iterable[int]) -> int:
    proposed = tuple(proposed)
    future = tuple(future)
    count = 0
    for a, b in zip(proposed, future):
        if a != b:
            break
        count += 1
    return count


def full_window_eligible(future: Iterable[int], eos: int) -> bool:
    future = tuple(future)
    return len(future) == 8 and eos not in future


def prompt_scores(rows: Iterable[Mapping], *, eligible_only: bool) -> dict[tuple, float]:
    """Return one score per (seed, source, family, prompt, arm)."""
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if eligible_only and not row["full_window_eligible"]:
            continue
        if not 0 <= row["acceptance"] <= min(8, row["future_length"]):
            raise ValueError("invalid acceptance or target horizon")
        key = (row["seed"], row["source"], row["family_id"],
               row["prompt_id"], row["arm"])
        grouped[key].append(float(row["acceptance"]))
    return {key: mean(values) for key, values in grouped.items()}


def _paired_prompt_differences(scores: Mapping[tuple, float], *, seed: int,
                               left: str, right: str) -> dict[str, dict[str, list[float]]]:
    base: dict[tuple, dict[str, float]] = defaultdict(dict)
    for (s, source, family, prompt, arm), value in scores.items():
        if s == seed and arm in (left, right):
            base[(source, family, prompt)][arm] = value
    family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (source, family_id, prompt), arms in base.items():
        if set(arms) != {left, right}:
            raise ValueError(f"unpaired prompt: {prompt}")
        family[source][family_id].append(arms[left] - arms[right])
    if not family:
        raise ValueError("no paired prompts")
    return family


def _source_balanced_mean(family: Mapping[str, Mapping[str, list[float]]],
                          source_weights: Mapping[str, int] | None = None) -> float:
    # A prompt is weighted equally.  During bootstrap, keep the *observed*
    # source weights fixed while resampling whole families inside each source.
    values = {source: [value for prompts in families.values()
                       for value in prompts]
              for source, families in family.items()}
    if not values or any(not item for item in values.values()):
        raise ValueError("empty paired source/family")
    weights = (source_weights if source_weights is not None else
               {source: len(item) for source, item in values.items()})
    if set(weights) != set(values) or any(weight <= 0 for weight in weights.values()):
        raise ValueError("source stratum weights differ")
    return sum(weights[source] * mean(item) for source, item in values.items()) / sum(weights.values())


def paired_family_ci(scores: Mapping[tuple, float], *, seed: int,
                     left: str, right: str, draws: int = BOOTSTRAP_DRAWS,
                     bootstrap_seed: int = BOOTSTRAP_SEED) -> dict:
    if draws <= 0:
        raise ValueError("draws must be positive")
    family = _paired_prompt_differences(scores, seed=seed, left=left, right=right)
    source_weights = {source: sum(len(prompts) for prompts in members.values())
                      for source, members in family.items()}
    observed = _source_balanced_mean(family, source_weights)
    source_observed = {source: _source_balanced_mean({source: members})
                       for source, members in family.items()}
    rng = random.Random(bootstrap_seed + seed * 1009)
    samples = []
    for _ in range(draws):
        resampled: dict[str, dict[str, list[float]]] = {}
        for source, members in family.items():
            keys = sorted(members)
            picks = [keys[rng.randrange(len(keys))] for _ in keys]
            # Preserve duplicate selected families as separate draws.
            resampled[source] = {str(i): members[key] for i, key in enumerate(picks)}
        samples.append(_source_balanced_mean(resampled, source_weights))
    samples.sort()
    lo = samples[int(0.025 * (draws - 1))]
    hi = samples[int(0.975 * (draws - 1))]
    return {
        "seed": seed, "comparison": f"{left}-{right}",
        "point": observed, "ci95": [lo, hi],
        "source_points": source_observed,
        "prompt_count": sum(len(prompts) for members in family.values()
                            for prompts in members.values()),
        "family_count": sum(map(len, family.values())),
        "source_prompt_weights": source_weights,
        "draws": draws, "bootstrap_seed": bootstrap_seed + seed * 1009,
        "conditional_on_training_seed": True,
    }
