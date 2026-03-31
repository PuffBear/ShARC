"""
Arc Encoder — Transformer over arc feature vectors.

Each arc (including depot at index 0) is treated as a node.
Arc features:
  service_time  scalar  (0 for depot)
  demand        scalar  (normalised, 0 for depot)
  clss          int {0,1,2,3} → learned embedding
  is_depot      binary flag

Output: contextualised arc embeddings [B, n+1, d_model]
"""

import torch
import torch.nn as nn


class ArcEncoder(nn.Module):
    """
    Transformer encoder over arc features.

    Parameters
    ----------
    d_model : int      hidden dimension
    n_heads : int      attention heads
    n_layers: int      number of TransformerEncoderLayers
    d_ff    : int      feed-forward inner dimension
    d_clss  : int      priority-class embedding dimension
    """

    N_CLASSES = 4  # priority classes: 0 (depot), 1, 2, 3

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        d_clss: int = 16,
    ):
        super().__init__()

        self.d_model = d_model
        self.clss_embed = nn.Embedding(self.N_CLASSES, d_clss)

        # feature dim: service_time(1) + demand(1) + clss_embed(d_clss) + is_depot(1)
        feat_dim = 3 + d_clss

        self.input_proj = nn.Linear(feat_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            dropout=0.0,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        service_time: torch.Tensor,   # [B, n+1]
        demand: torch.Tensor,         # [B, n+1]
        clss: torch.Tensor,           # [B, n+1]  long
    ) -> torch.Tensor:
        """
        Returns arc embeddings [B, n+1, d_model].
        """
        B, n1 = service_time.shape
        device = service_time.device

        is_depot = torch.zeros(B, n1, 1, device=device)
        is_depot[:, 0] = 1.0

        clss_emb = self.clss_embed(clss)   # [B, n+1, d_clss]

        x = torch.cat(
            [
                service_time.unsqueeze(-1),  # [B, n+1, 1]
                demand.unsqueeze(-1),         # [B, n+1, 1]
                clss_emb,                     # [B, n+1, d_clss]
                is_depot,                     # [B, n+1, 1]
            ],
            dim=-1,
        )  # [B, n+1, feat_dim]

        x = self.input_proj(x)            # [B, n+1, d_model]
        return self.transformer(x)        # [B, n+1, d_model]
