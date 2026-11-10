"""Same-worker teacher-consumption diagnostic before costly v11 collection.

Wrong-family and zero features are explicit diagnostic interventions. They
never enter the training corpus or get represented as valid Target provenance.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .v10_four_arm import V10FourArmHead, V10PreparedState


def tensor_sha(t: torch.Tensor) -> str:
    x = t.detach().contiguous().cpu()
    return hashlib.sha256(str(x.dtype).encode() + str(tuple(x.shape)).encode()
                          + x.view(torch.uint8).numpy().tobytes()).hexdigest()


def _control_and_repair(head: V10FourArmHead, state: V10PreparedState,
                        draft_hidden: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    core = head.core
    repair = core.repair
    local = state.local_hidden
    codes = core.local_encode(local, state.rejected_embedding,
                              state.correction_embedding, state.accepted_count)
    code_control = repair._lookup(codes[:, 0], 1) + repair._lookup(codes[:, 1], 1)
    surrogate = head.local_hidden_lift(local.to(head.local_hidden_lift[0].weight.dtype))
    if state.target_hidden is not None:
        surrogate = surrogate + state.target_hidden.to(
            device=surrogate.device, dtype=surrogate.dtype)
    hidden_control = head.hidden_projection(surrogate)
    semantic = head.local_top32_surrogate(local.to(head.local_top32_surrogate.weight.dtype))
    if state.target_top32_features is not None:
        semantic = semantic + state.target_top32_features.to(
            device=semantic.device, dtype=semantic.dtype)
    top_control = head.top32_projection(semantic)
    control = code_control + hidden_control + top_control
    u = repair.V(draft_hidden.to(repair.V.weight.dtype))
    pos = torch.ones((1, 1), dtype=u.dtype, device=u.device) / head.config.window
    gate = repair.gate_mlp(torch.cat((control.to(u.dtype), u, pos), -1))
    residual_before_bf16 = repair.U(gate * u)
    return {"code": code_control, "hidden": hidden_control,
            "top32": top_control, "total": control}, residual_before_bf16


def _max_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def run_teacher_diagnostic(*, heads: dict[str, V10FourArmHead],
                           windows: Sequence[Any], draft: Any) -> dict[str, Any]:
    if len(windows) != 8 or set(heads) != {"B", "C", "F-H", "F-CURRENT-TOP32"}:
        raise ValueError("v11 diagnostic requires v10 four heads and first eight windows")
    receipt: dict[str, Any] = {
        "scope": "v11_same_worker_precollection_teacher_input_consumption",
        "status": "running", "windows_required": 8,
        "wrong_donor_rule": "same_source_different_prompt_and_family_or_NA",
        "wrong_and_zero_are_diagnostic_only": True,
        "arms": {},
    }
    for mode, head in heads.items():
        head.eval()
        records = []
        for item in windows:
            base_obs = item.observation
            donor = next((other for other in windows
                          if other.observation.prompt_id != base_obs.prompt_id
                          and other.observation.family_id != base_obs.family_id
                          and other.observation.task_type == base_obs.task_type), None)
            donor_obs = donor.observation if donor is not None else None
            hidden = draft.teacher_hidden(list(base_obs.corrected_prefix), [0]).detach()
            correct = head.prepare_observation(base_obs, draft)
            if mode in ("B", "C"):
                zero = head.prepare_observation(replace(
                    base_obs, target_hidden=torch.zeros_like(base_obs.target_hidden)), draft)
                wrong = (head.prepare_observation(replace(
                    base_obs, target_hidden=donor_obs.target_hidden,
                    target_logit_summary=donor_obs.target_logit_summary), draft)
                         if donor_obs is not None else None)
            elif mode == "F-H":
                feature = correct.target_hidden
                if feature is None:
                    raise RuntimeError("F-H teacher hidden not prepared")
                zero = replace(correct, target_hidden=torch.zeros_like(feature))
                wrong = (replace(correct, target_hidden=donor_obs.target_hidden
                         .reshape(1, -1).to(feature.device, feature.dtype))
                         if donor_obs is not None else None)
            else:
                feature = correct.target_top32_features
                if feature is None:
                    raise RuntimeError("TOP32 teacher summary not prepared")
                zero = replace(correct, target_top32_features=torch.zeros_like(feature))
                wrong = (replace(correct, target_top32_features=
                         head.prepare_observation(donor_obs, draft).target_top32_features)
                         if donor_obs is not None else None)
            with torch.no_grad():
                outputs = {}
                for name, state in (("correct", correct), ("zero", zero), ("wrong", wrong)):
                    if state is None:
                        continue
                    control, residual = _control_and_repair(head, state, hidden)
                    logits = head.forward_logits(hidden, state, 1)
                    outputs[name] = {
                        "control": control, "residual": residual, "logits": logits,
                        "feature_sha256": tensor_sha(
                            state.target_hidden if mode == "F-H" else
                            state.target_top32_features) if mode.startswith("F-") else None,
                    }
            reference = outputs["correct"]
            deltas = {other: {
                "control_max_abs": _max_delta(reference["control"]["total"],
                                              outputs[other]["control"]["total"]),
                "control_component_max_abs": {
                    key: _max_delta(reference["control"][key],
                                    outputs[other]["control"][key])
                    for key in reference["control"]},
                "repair_residual_prequant_max_abs": _max_delta(
                    reference["residual"], outputs[other]["residual"]),
                "bf16_logits_max_abs": _max_delta(reference["logits"], outputs[other]["logits"]),
            } for other in outputs if other != "correct"}
            gradient = None
            if mode.startswith("F-"):
                feature = (correct.target_hidden if mode == "F-H" else
                           correct.target_top32_features).detach().clone().requires_grad_(True)
                state = (replace(correct, target_hidden=feature) if mode == "F-H"
                         else replace(correct, target_top32_features=feature))
                head.zero_grad(set_to_none=True)
                logits = head.forward_logits(hidden, state, 1)
                label = torch.tensor([item.labels[0]], dtype=torch.long, device=logits.device)
                F.cross_entropy(logits.float(), label).backward()
                gradient = (float(feature.grad.detach().float().norm().item())
                            if feature.grad is not None else 0.0)
                head.zero_grad(set_to_none=True)
            if any(not math.isfinite(value)
                   for d in deltas.values()
                   for value in (d["control_max_abs"],
                                 d["repair_residual_prequant_max_abs"],
                                 d["bf16_logits_max_abs"],
                                 *d["control_component_max_abs"].values())):
                raise FloatingPointError("nonfinite teacher diagnostic delta")
            protocol_payload = {
                "valid_prefix": base_obs.valid_prefix,
                "accepted_count": base_obs.accepted_count,
                "rejected_token": base_obs.rejected_token,
                "correction_token": base_obs.correction_token,
            }
            records.append({
                "prompt_id": base_obs.prompt_id,
                "round_index": base_obs.round_index,
                "source": base_obs.task_type,
                "donor_prompt_id": donor_obs.prompt_id if donor_obs else None,
                "wrong_donor_NA": donor_obs is None,
                "local_hidden_sha256": tensor_sha(correct.local_hidden),
                "protocol_sha256": hashlib.sha256(json.dumps(
                    protocol_payload, sort_keys=True).encode()).hexdigest(),
                "correct_feature_sha256": reference["feature_sha256"],
                "zero_feature_sha256": outputs["zero"]["feature_sha256"],
                "wrong_feature_sha256": (outputs["wrong"]["feature_sha256"]
                                         if "wrong" in outputs else None),
                "deltas": deltas,
                "correct_control_component_norms": {
                    key: float(value.detach().float().norm().item())
                    for key, value in reference["control"].items()},
                "correct_repair_residual_prequant_norm": float(
                    reference["residual"].detach().float().norm().item()),
                "teacher_feature_leaf_ce_gradient_norm": gradient,
            })
        if mode in ("B", "C"):
            passed = all(all(value == 0 for value in (
                             delta["control_max_abs"],
                             delta["repair_residual_prequant_max_abs"],
                             delta["bf16_logits_max_abs"],
                             *delta["control_component_max_abs"].values()))
                         for record in records for delta in record["deltas"].values())
        else:
            passed = (any(record["deltas"]["zero"]["control_max_abs"] > 0 and
                          record["deltas"]["zero"]["repair_residual_prequant_max_abs"] > 0 and
                          record["teacher_feature_leaf_ce_gradient_norm"] > 0
                          for record in records)
                      and any("wrong" in record["deltas"] and
                              record["deltas"]["wrong"]["control_max_abs"] > 0
                              for record in records))
        receipt["arms"][mode] = {
            "window_count": len(records), "wrong_donor_NA_count": sum(
                record["wrong_donor_NA"] for record in records),
            "windows": records, "passed": passed,
        }
        if not passed:
            receipt["status"] = "failed"
            return receipt
    receipt["status"] = "passed"
    return receipt
