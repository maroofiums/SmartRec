import torch
import torch.nn as nn
from typing import Optional


class NeuralMF(nn.Module):
    """
    Hybrid Neural Matrix Factorization model.

    Fuses two signals:
    - Collaborative: learned user & item embeddings from interaction history
    - Content: item metadata encoded by a small MLP

    Final scoring head takes the concatenation of all three vectors.

    Forward pass:
        user_ids   : (B,)         — integer user indices
        item_ids   : (B,)         — integer item indices
        content    : (B, C)       — content feature vector (genres + tags encoded)

    Returns:
        scores     : (B,)         — predicted relevance in [0, 1]
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        content_dim: int = 32,
        hidden_dims: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Collaborative embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)

        # Content encoder — projects raw content features → embedding_dim
        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Fusion MLP — input: [user_emb || item_emb || content_emb]
        input_dim = embedding_dim * 3
        layers: list[nn.Module] = []
        for hdim in hidden_dims:
            layers += [nn.Linear(input_dim, hdim), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = hdim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        content: torch.Tensor,
    ) -> torch.Tensor:
        u_emb = self.user_embedding(user_ids)            # (B, D)
        i_emb = self.item_embedding(item_ids)            # (B, D)
        c_emb = self.content_encoder(content)            # (B, D)

        fused = torch.cat([u_emb, i_emb, c_emb], dim=-1)  # (B, 3D)
        score = self.mlp(fused).squeeze(-1)              # (B,)
        return score

    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        idx = torch.tensor([user_id])
        return self.user_embedding(idx).detach()

    def get_item_embedding(self, item_id: int, content: torch.Tensor) -> torch.Tensor:
        idx = torch.tensor([item_id])
        i_emb = self.item_embedding(idx)
        c_emb = self.content_encoder(content.unsqueeze(0))
        return torch.cat([i_emb, c_emb], dim=-1).detach()


class BPRLoss(nn.Module):
    """
    Bayesian Personalised Ranking loss for implicit feedback.
    Maximises the margin between positive and negative item scores.
    """

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
