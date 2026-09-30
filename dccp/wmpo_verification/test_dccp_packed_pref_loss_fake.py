"""Fake packed-preference tests for DCCP local DPO-style loss.

These tests do not load OpenVLA or a real VLA checkpoint. They instantiate the
actor object without calling its heavy __init__ and monkeypatch sequence-logprob
computation so the packed preference interface can be tested cheaply.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.workers.actor.dp_rob import RobDataParallelPPOActor
from verl.utils.dccp_schema import PREF_KEYS


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def make_actor(use_ref_gap=True):
    actor = RobDataParallelPPOActor.__new__(RobDataParallelPPOActor)
    actor.config = AttrDict(
        {
            "use_dccp_branch": True,
            "vla": "openvla-oft",
            "dccp": {
                "enable_loss": True,
                "beta": 0.5,
                "lambda_pref": 0.3,
                "lambda_pref_warmup_steps": 0,
                "use_ref_gap": use_ref_gap,
                "packed_pref_interface": True,
            },
        }
    )
    actor.pad_token_id = 0

    def fake_logprob(self, input_ids, attention_mask, pixel_values, responses, response_mask, temperature):
        del input_ids, attention_mask, pixel_values, temperature
        values = responses.float().reshape(responses.shape[0], -1).sum(dim=-1)
        mask_scale = response_mask.float().reshape(response_mask.shape[0], -1).mean(dim=-1)
        return values * mask_scale / 100.0

    actor._forward_dccp_sequence_logprob = MethodType(fake_logprob, actor)
    return actor


def make_batch(use_alias=False, all_invalid=False):
    batch_size = 2
    pairs = 2
    prompt_len = 5
    action_len = 3

    batch = {
        PREF_KEYS.input_ids: torch.arange(batch_size * pairs * prompt_len).reshape(batch_size, pairs, prompt_len),
        PREF_KEYS.attention_mask: torch.ones(batch_size, pairs, prompt_len, dtype=torch.long),
        PREF_KEYS.pixel_values: torch.zeros(batch_size, pairs, 3, 8, 8),
        PREF_KEYS.response_mask: torch.ones(batch_size, pairs, action_len, dtype=torch.bool),
        PREF_KEYS.weight: torch.tensor([[1.0, 0.5], [0.25, 2.0]]),
        PREF_KEYS.delta_ref: torch.tensor([[0.1, -0.2], [0.0, 0.3]]),
        PREF_KEYS.margin: torch.tensor([[0.2, -0.4], [0.0, 0.8]]),
        PREF_KEYS.valid: torch.zeros(batch_size, pairs, dtype=torch.bool)
        if all_invalid
        else torch.tensor([[True, False], [True, True]]),
        "responses": torch.ones(batch_size, 1, action_len, dtype=torch.long),
        "input_ids": torch.ones(batch_size, 1, prompt_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 1, prompt_len, dtype=torch.long),
        "pixel_values": torch.zeros(batch_size, 1, 3, 8, 8),
    }

    winner = torch.tensor(
        [
            [[10, 11, 12], [13, 14, 15]],
            [[16, 17, 18], [19, 20, 21]],
        ],
        dtype=torch.long,
    )
    loser = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=torch.long,
    )

    if use_alias:
        batch["pref_aw_responses"] = winner
        batch["pref_al_responses"] = loser
    else:
        batch[PREF_KEYS.winner_responses] = winner
        batch[PREF_KEYS.loser_responses] = loser

    return batch


def test_all_invalid_pref_valid_returns_zero_loss():
    actor = make_actor()
    loss, metrics = actor._compute_dccp_pref_loss(make_batch(all_invalid=True), temperature=1.0)

    assert loss.ndim == 0
    assert torch.isclose(loss.detach(), torch.tensor(0.0))
    assert metrics["dccp/valid_pairs"] == 0.0
    assert metrics["loss/dccp_pref"] == 0.0


def test_packed_winner_loser_responses_loss_is_scalar():
    actor = make_actor()
    loss, metrics = actor._compute_dccp_pref_loss(make_batch(), temperature=1.0)

    assert loss.ndim == 0
    assert torch.isfinite(loss.detach())
    assert metrics["dccp/valid_pairs"] == 3.0
    assert metrics["loss/dccp_pref"] > 0.0
    assert "dccp/delta_theta_mean" in metrics


def test_aw_al_response_aliases_are_supported():
    actor = make_actor()
    loss, metrics = actor._compute_dccp_pref_loss(make_batch(use_alias=True), temperature=1.0)

    assert loss.ndim == 0
    assert torch.isfinite(loss.detach())
    assert metrics["dccp/valid_pairs"] == 3.0


def test_use_ref_gap_false_ignores_pref_delta_ref():
    batch = make_batch()
    actor = make_actor(use_ref_gap=False)
    loss_without_ref, metrics_without_ref = actor._compute_dccp_pref_loss(batch, temperature=1.0)

    batch_shifted_ref = make_batch()
    batch_shifted_ref[PREF_KEYS.delta_ref] = batch_shifted_ref[PREF_KEYS.delta_ref] + 1000.0
    loss_shifted_ref, metrics_shifted_ref = actor._compute_dccp_pref_loss(batch_shifted_ref, temperature=1.0)

    assert torch.allclose(loss_without_ref.detach(), loss_shifted_ref.detach())
    assert metrics_without_ref["dccp/use_ref_gap"] == 0.0
    assert metrics_shifted_ref["dccp/use_ref_gap"] == 0.0


if __name__ == "__main__":
    test_all_invalid_pref_valid_returns_zero_loss()
    test_packed_winner_loser_responses_loss_is_scalar()
    test_aw_al_response_aliases_are_supported()
    test_use_ref_gap_false_ignores_pref_delta_ref()
    print("DCCP packed preference fake tests passed")
