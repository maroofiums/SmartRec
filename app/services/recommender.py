import torch
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.models.neural_mf import NeuralMF
from app.models.orm import User, Item, UserEmbeddingCache, ItemEmbeddingCache
from app.schemas.schemas import RecommendedItem
from app.core.config import get_settings

settings = get_settings()


class RecommenderService:
    """
    Wraps the trained NeuralMF model and handles:
    - Loading/saving the model checkpoint
    - Embedding cache read/write
    - Top-K inference for a given user
    """

    def __init__(self):
        self.model: Optional[NeuralMF] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, num_users: int, num_items: int, path: str = None):
        path = path or settings.model_path
        self.model = NeuralMF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=settings.embedding_dim,
            hidden_dims=settings.hidden_dims,
            dropout=settings.dropout,
        ).to(self.device)

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _content_tensor(self, item: Item, content_dim: int = 32) -> torch.Tensor:
        """
        Build a fixed-size content vector from an Item row.
        Uses the pre-stored content_vector if available, otherwise zeros.
        """
        if item.content_vector and len(item.content_vector) > 0:
            vec = np.array(item.content_vector, dtype=np.float32)
        else:
            vec = np.zeros(content_dim, dtype=np.float32)

        # Pad or truncate to content_dim
        if len(vec) < content_dim:
            vec = np.pad(vec, (0, content_dim - len(vec)))
        else:
            vec = vec[:content_dim]

        return torch.tensor(vec, dtype=torch.float32).to(self.device)

    async def get_recommendations(
        self,
        user: User,
        db: AsyncSession,
        top_k: int = None,
    ) -> list[RecommendedItem]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        top_k = top_k or settings.top_k

        # Fetch all items
        result = await db.execute(select(Item))
        items = result.scalars().all()
        if not items:
            return []

        # Try to get user embedding from cache
        cached = await db.execute(
            select(UserEmbeddingCache).where(UserEmbeddingCache.user_id == user.id)
        )
        user_cache = cached.scalar_one_or_none()

        with torch.no_grad():
            if user_cache:
                u_emb = torch.tensor(user_cache.embedding, dtype=torch.float32).to(self.device)
            else:
                u_emb = self.model.user_embedding(
                    torch.tensor([user.id], device=self.device)
                ).squeeze(0)

            scores: list[tuple[float, Item]] = []

            for item in items:
                # Try item embedding cache
                item_cached = await db.execute(
                    select(ItemEmbeddingCache).where(ItemEmbeddingCache.item_id == item.id)
                )
                item_cache = item_cached.scalar_one_or_none()

                content = self._content_tensor(item)

                if item_cache:
                    i_emb = torch.tensor(item_cache.embedding, dtype=torch.float32).to(self.device)
                else:
                    i_emb_cf = self.model.item_embedding(
                        torch.tensor([item.id], device=self.device)
                    ).squeeze(0)
                    i_emb_c = self.model.content_encoder(content)
                    i_emb = torch.cat([i_emb_cf, i_emb_c], dim=-1)

                # Score via fusion MLP
                fused = torch.cat([u_emb, i_emb[:settings.embedding_dim], content]).unsqueeze(0)

                # Re-use full forward pass for correctness
                score = self.model(
                    torch.tensor([user.id], device=self.device),
                    torch.tensor([item.id], device=self.device),
                    content.unsqueeze(0),
                ).item()

                scores.append((score, item))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]

        return [
            RecommendedItem(
                item_id=item.id,
                external_id=item.external_id,
                title=item.title,
                genres=item.genres or [],
                score=round(score, 4),
            )
            for score, item in top
        ]

    async def cache_user_embedding(self, user: User, db: AsyncSession):
        """Store user embedding in PostgreSQL to skip re-inference next time."""
        with torch.no_grad():
            emb = self.model.user_embedding(
                torch.tensor([user.id], device=self.device)
            ).squeeze(0).cpu().tolist()

        existing = await db.execute(
            select(UserEmbeddingCache).where(UserEmbeddingCache.user_id == user.id)
        )
        cache = existing.scalar_one_or_none()

        if cache:
            cache.embedding = emb
        else:
            db.add(UserEmbeddingCache(user_id=user.id, embedding=emb))

        await db.commit()

    async def cache_all_item_embeddings(self, db: AsyncSession):
        """Pre-compute and store all item embeddings. Run after training."""
        result = await db.execute(select(Item))
        items = result.scalars().all()

        for item in items:
            content = self._content_tensor(item)
            with torch.no_grad():
                i_emb_cf = self.model.item_embedding(
                    torch.tensor([item.id], device=self.device)
                ).squeeze(0)
                i_emb_c = self.model.content_encoder(content)
                combined = torch.cat([i_emb_cf, i_emb_c], dim=-1).cpu().tolist()

            existing = await db.execute(
                select(ItemEmbeddingCache).where(ItemEmbeddingCache.item_id == item.id)
            )
            cache = existing.scalar_one_or_none()
            if cache:
                cache.embedding = combined
            else:
                db.add(ItemEmbeddingCache(item_id=item.id, embedding=combined))

        await db.commit()


# Singleton — loaded once at startup
recommender = RecommenderService()
