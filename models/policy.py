"""
HCARPPolicy — full encoder-decoder policy for HCARP.

The policy supports:
  - Budget conditioning: the decoder context includes b_t (running max vehicle time),
    enabling the policy to reason about its worst-case cost trajectory.
  - Shift context conditioning (d_shift > 0): a projected shift context vector
    [delta_demand, delta_cost, p_availability] is appended to the decoder context,
    allowing the policy to adapt to the current distribution shift.

Usage
-----
policy = HCARPPolicy()

# Once per episode: encode static instance features
arc_emb = policy.encode(obs)          # [B, n+1, d_model]

# Each decoding step: sample action and record log-prob
action, log_p = policy.act(obs, arc_emb)
action, log_p = policy.act(obs, arc_emb, greedy=True)

# Full episode rollout (returns sequences for REINFORCE)
actions, log_probs, rewards, info = policy.rollout(env)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.encoder import ArcEncoder
from models.decoder import AttentionDecoder


def _to_tensor(x, dtype=torch.float32, device="cpu"):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    return x.to(dtype=dtype, device=device)


def _to_long(x, device="cpu"):
    return _to_tensor(x, dtype=torch.long, device=device)


def _to_bool(x, device="cpu"):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=torch.bool, device=device)
    return x.to(dtype=torch.bool, device=device)


class HCARPPolicy(nn.Module):
    """
    Attention-based policy for HCARP with budget and optional shift conditioning.

    Parameters
    ----------
    d_model       : hidden dimension throughout
    n_heads       : attention heads (encoder + decoder)
    n_enc_layers  : number of Transformer encoder layers
    d_ff          : feed-forward inner dimension
    d_clss        : priority-class embedding dimension
    clip          : tanh clipping range for pointer logits
    d_shift       : projected shift context dimension (0 = no shift conditioning)
    device        : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_enc_layers: int = 3,
        d_ff: int = 512,
        d_clss: int = 16,
        clip: float = 10.0,
        d_shift: int = 8,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device
        self.d_shift = d_shift

        self.encoder = ArcEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            d_ff=d_ff,
            d_clss=d_clss,
        )
        self.decoder = AttentionDecoder(
            d_model=d_model,
            n_heads=n_heads,
            clip=clip,
            d_shift=d_shift,
        )

        if d_shift > 0:
            # Projects the 3-dim shift context into d_shift dims
            self.shift_proj = nn.Linear(3, d_shift)
        else:
            self.shift_proj = None

        self.to(device)

    # ------------------------------------------------------------------
    # Encode  (call once per episode)
    # ------------------------------------------------------------------

    def encode(self, obs: dict) -> torch.Tensor:
        """
        Encode static arc features.

        Returns
        -------
        arc_emb : [B, n+1, d_model]
        """
        dev = self.device
        service_time = _to_tensor(obs["service_time"], device=dev)
        demand       = _to_tensor(obs["demand"],       device=dev)
        clss         = _to_long(obs["clss"],           device=dev)

        return self.encoder(service_time, demand, clss)  # [B, n+1, d_model]

    # ------------------------------------------------------------------
    # Act  (call each step)
    # ------------------------------------------------------------------

    def act(
        self,
        obs: dict,
        arc_emb: torch.Tensor,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select one action per instance given current observation.

        Returns
        -------
        actions   : LongTensor  [B]
        log_probs : FloatTensor [B]
        """
        dev = self.device
        B = arc_emb.size(0)

        cur_arc       = _to_long(obs["cur_arc"],         device=dev)  # [B, M]
        remaining_cap = _to_tensor(obs["remaining_cap"], device=dev)  # [B, M]
        vehicle_time  = _to_tensor(obs["vehicle_time"],  device=dev)  # [B, M]
        active_v      = _to_long(obs["active_vehicle"],  device=dev)  # [B]
        mask          = _to_bool(obs["action_mask"],     device=dev)  # [B, n+1]
        budget        = _to_tensor(obs["budget"],        device=dev)  # [B]

        # Gather active vehicle's state
        av_idx = active_v.unsqueeze(1)
        cur_v  = cur_arc.gather(1, av_idx).squeeze(1)        # [B]
        cap_v  = remaining_cap.gather(1, av_idx).squeeze(1)  # [B]
        time_v = vehicle_time.gather(1, av_idx).squeeze(1)   # [B]

        cur_emb = arc_emb[torch.arange(B, device=dev), cur_v]  # [B, d_model]

        # Build context: [cur_arc_emb | cap | time | budget | (shift_emb)]
        context_parts = [
            cur_emb,
            cap_v.unsqueeze(-1),
            time_v.unsqueeze(-1),
            budget.unsqueeze(-1),
        ]

        if self.d_shift > 0:
            raw_shift = _to_tensor(
                obs.get("shift_context", np.zeros((B, 3), dtype=np.float32)),
                device=dev,
            )  # [B, 3]
            shift_emb = self.shift_proj(raw_shift)  # [B, d_shift]
            context_parts.append(shift_emb)

        context = torch.cat(context_parts, dim=-1)  # [B, d_model + 3 + d_shift]

        log_probs_all = self.decoder(arc_emb, context, mask)  # [B, n+1]

        if greedy:
            actions = log_probs_all.argmax(dim=-1)
        else:
            probs = log_probs_all.exp()
            actions = torch.distributions.Categorical(probs=probs).sample()

        selected_log_p = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        return actions, selected_log_p

    # ------------------------------------------------------------------
    # Full episode rollout
    # ------------------------------------------------------------------

    def rollout(
        self,
        env,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Roll out one complete episode on `env` (which must already be reset).

        Returns
        -------
        actions   : LongTensor  [B, T]
        log_probs : FloatTensor [B, T]
        rewards   : FloatTensor [B]
        info      : dict
        """
        obs = env._get_obs()
        arc_emb = self.encode(obs)

        all_actions: list[torch.Tensor] = []
        all_log_p:   list[torch.Tensor] = []

        rewards = torch.zeros(env.B, device=self.device)
        final_info: dict = {}

        for _ in range(env.max_steps() + env.M):
            actions, log_p = self.act(obs, arc_emb, greedy=greedy)

            actions_np = actions.cpu().numpy().astype(np.int32)
            obs, reward_np, done_np, info = env.step(actions_np)

            all_actions.append(actions)
            all_log_p.append(log_p)

            rewards += torch.from_numpy(reward_np).to(self.device)
            final_info.update(info)

            if done_np.all():
                break

        actions_seq  = torch.stack(all_actions, dim=1)   # [B, T]
        log_prob_seq = torch.stack(all_log_p,   dim=1)   # [B, T]

        return actions_seq, log_prob_seq, rewards, final_info
