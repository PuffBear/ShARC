"""
HCARPPolicy — full encoder-decoder policy for HCARP.

Usage
-----
policy = HCARPPolicy()

# Once per episode: encode static instance features
arc_emb = policy.encode(obs)          # [B, n+1, d_model]

# Each decoding step: sample action and record log-prob
action, log_p = policy.act(obs, arc_emb)          # stochastic
action, log_p = policy.act(obs, arc_emb, greedy=True)  # deterministic

# Full episode rollout (returns sequences for REINFORCE)
actions, log_probs = policy.rollout(env)
"""

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
    Attention-based policy for HCARP.

    Parameters
    ----------
    d_model       : hidden dimension throughout
    n_heads       : attention heads (encoder + decoder)
    n_enc_layers  : number of Transformer encoder layers
    d_ff          : feed-forward inner dimension
    d_clss        : priority-class embedding dimension
    clip          : tanh clipping range for pointer logits
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
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device

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
        )

        self.to(device)

    # ------------------------------------------------------------------
    # Encode  (call once per episode)
    # ------------------------------------------------------------------

    def encode(self, obs: dict) -> torch.Tensor:
        """
        Encode static arc features.

        Parameters
        ----------
        obs : observation dict from HCARPEnv (numpy arrays or tensors)

        Returns
        -------
        arc_emb : [B, n+1, d_model]
        """
        dev = self.device
        service_time = _to_tensor(obs["service_time"], device=dev)  # [B, n+1]
        demand       = _to_tensor(obs["demand"],       device=dev)  # [B, n+1]
        clss         = _to_long(obs["clss"],           device=dev)  # [B, n+1]

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

        Parameters
        ----------
        obs     : current observation dict
        arc_emb : [B, n+1, d_model] from encode()
        greedy  : if True use argmax, else sample

        Returns
        -------
        actions   : LongTensor  [B]
        log_probs : FloatTensor [B]  — log prob of the selected action
        """
        dev = self.device
        B = arc_emb.size(0)

        cur_arc       = _to_long(obs["cur_arc"],       device=dev)  # [B, M]
        remaining_cap = _to_tensor(obs["remaining_cap"], device=dev) # [B, M]
        vehicle_time  = _to_tensor(obs["vehicle_time"], device=dev)  # [B, M]
        active_v      = _to_long(obs["active_vehicle"], device=dev)  # [B]
        mask          = _to_bool(obs["action_mask"],    device=dev)  # [B, n+1]

        # Gather active vehicle's state for each instance in the batch
        av_idx = active_v.unsqueeze(1)                      # [B, 1]
        cur_v  = cur_arc.gather(1, av_idx).squeeze(1)       # [B]
        cap_v  = remaining_cap.gather(1, av_idx).squeeze(1) # [B]
        time_v = vehicle_time.gather(1, av_idx).squeeze(1)  # [B]

        # Embedding of the active vehicle's current arc
        cur_emb = arc_emb[torch.arange(B, device=dev), cur_v]  # [B, d_model]

        # Decoder context: [cur_arc_emb | remaining_cap | vehicle_time]
        context = torch.cat(
            [cur_emb, cap_v.unsqueeze(-1), time_v.unsqueeze(-1)], dim=-1
        )  # [B, d_model + 2]

        log_probs_all = self.decoder(arc_emb, context, mask)  # [B, n+1]

        if greedy:
            actions = log_probs_all.argmax(dim=-1)  # [B]
        else:
            # log_probs_all contains log-probabilities — convert to probs for sampling.
            # (Categorical(logits=...) would double-apply softmax and give wrong distribution.)
            probs = log_probs_all.exp()
            actions = torch.distributions.Categorical(probs=probs).sample()

        # Log-prob of the chosen action
        selected_log_p = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        return actions, selected_log_p

    # ------------------------------------------------------------------
    # Full episode rollout  (convenience for REINFORCE training)
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
        actions   : LongTensor  [B, T]      — action at each step
        log_probs : FloatTensor [B, T]      — log-prob at each step
        rewards   : FloatTensor [B]         — final reward per instance
        info      : dict                    — T1/T2/T3 per instance
        """
        obs = env._get_obs()
        arc_emb = self.encode(obs)

        all_actions: list[torch.Tensor] = []
        all_log_p:   list[torch.Tensor] = []

        rewards = torch.zeros(env.B, device=self.device)
        final_info: dict = {}

        for _ in range(env.max_steps() + env.M):  # upper bound on steps
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
