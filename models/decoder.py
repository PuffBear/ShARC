"""
Attention Decoder — single decoding step for HCARP.

At each step the decoder receives:
  - arc_emb   [B, n+1, d_model]  from the encoder (fixed for the episode)
  - context   [B, d_model + 2]   [cur_arc_embedding, remaining_cap, vehicle_time]
  - mask      [B, n+1]  bool     True = feasible action

It produces a log-probability distribution over arcs [B, n+1].

Design
------
1. Project context → d_model query vector via a linear layer.
2. Refine query with one round of multi-head cross-attention over arc embeddings.
3. Compute pointer logits via scaled dot-product with a tanh clip (clip=10).
4. Mask infeasible actions and return log-softmax probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDecoder(nn.Module):
    """
    Parameters
    ----------
    d_model : int   must match the encoder's d_model
    n_heads : int   attention heads
    clip    : float tanh clipping range for pointer logits
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, clip: float = 10.0):
        super().__init__()

        self.d_model = d_model
        self.clip = clip

        # Context has d_model (arc emb) + 1 (cap) + 1 (time)
        self.ctx_proj = nn.Linear(d_model + 2, d_model)

        # Cross-attention: query = context, key/value = arc embeddings
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0
        )

        # Pointer projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        arc_emb: torch.Tensor,   # [B, n+1, d_model]
        context: torch.Tensor,   # [B, d_model + 2]
        mask: torch.Tensor,      # [B, n+1]  bool, True = feasible
    ) -> torch.Tensor:
        """
        Returns log_probs [B, n+1].
        Infeasible positions have log_prob = -inf.
        """
        # 1. Project context to query vector
        h = self.ctx_proj(context).unsqueeze(1)  # [B, 1, d_model]

        # 2. Cross-attention: refine query using arc embeddings
        #    key_padding_mask: True = IGNORE that position
        #    We want to attend only to feasible arcs during this step
        attn_mask = ~mask  # [B, n+1]  True = ignore
        h, _ = self.cross_attn(
            query=h,
            key=arc_emb,
            value=arc_emb,
            key_padding_mask=attn_mask,
        )  # [B, 1, d_model]
        h = h.squeeze(1)  # [B, d_model]

        # 3. Pointer: scaled dot-product with tanh clipping
        q = self.W_q(h)         # [B, d_model]
        k = self.W_k(arc_emb)   # [B, n+1, d_model]

        logits = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)   # [B, n+1]
        logits = logits / (self.d_model ** 0.5)
        logits = self.clip * torch.tanh(logits)

        # 4. Mask and return log-softmax
        logits = logits.masked_fill(~mask, float("-inf"))
        return F.log_softmax(logits, dim=-1)  # [B, n+1]
