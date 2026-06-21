"""
Training entrypoint for SmartRec Neural MF model.

Usage:
    python scripts/train.py [--epochs 20] [--batch-size 256] [--lr 1e-3]

Expects the database to be seeded first:
    python scripts/seed_data.py
"""

import asyncio
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import select

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.db.session import AsyncSessionLocal, init_db
from app.models.orm import User, Item, Interaction
from app.models.neural_mf import NeuralMF, BPRLoss
from app.core.config import get_settings

settings = get_settings()


class InteractionDataset(Dataset):
    """
    Implicit feedback dataset with BPR negative sampling.
    Each sample: (user_id, pos_item_id, neg_item_id, content_pos, content_neg)
    """

    def __init__(self, interactions, all_item_ids, item_content_map, content_dim=32):
        self.interactions = interactions
        self.all_item_ids = set(all_item_ids)
        self.item_content_map = item_content_map
        self.content_dim = content_dim

        # Build user -> positive items set for negative sampling
        self.user_pos: dict[int, set[int]] = {}
        for uid, iid in interactions:
            self.user_pos.setdefault(uid, set()).add(iid)

    def _content(self, item_id: int) -> np.ndarray:
        vec = self.item_content_map.get(item_id, [])
        arr = np.array(vec, dtype=np.float32) if vec else np.zeros(self.content_dim, dtype=np.float32)
        if len(arr) < self.content_dim:
            arr = np.pad(arr, (0, self.content_dim - len(arr)))
        return arr[:self.content_dim]

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_id, pos_item_id = self.interactions[idx]

        # Sample a negative item the user hasn't interacted with
        neg_item_id = random.choice(list(self.all_item_ids - self.user_pos.get(user_id, set())))

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "pos_item_id": torch.tensor(pos_item_id, dtype=torch.long),
            "neg_item_id": torch.tensor(neg_item_id, dtype=torch.long),
            "content_pos": torch.tensor(self._content(pos_item_id), dtype=torch.float32),
            "content_neg": torch.tensor(self._content(neg_item_id), dtype=torch.float32),
        }


async def load_data():
    async with AsyncSessionLocal() as db:
        users = (await db.execute(select(User))).scalars().all()
        items = (await db.execute(select(Item))).scalars().all()
        interactions = (await db.execute(select(Interaction))).scalars().all()

    user_ids = [u.id for u in users]
    item_ids = [i.id for i in items]
    item_content_map = {i.id: i.content_vector for i in items}
    pairs = [(ix.user_id, ix.item_id) for ix in interactions]

    return user_ids, item_ids, item_content_map, pairs


def ndcg_at_k(scores, labels, k=10):
    """Compute NDCG@K for a single user."""
    top_k_idx = np.argsort(scores)[::-1][:k]
    dcg = sum(labels[i] / np.log2(rank + 2) for rank, i in enumerate(top_k_idx))
    ideal = sum(1 / np.log2(rank + 2) for rank in range(min(int(labels.sum()), k)))
    return dcg / ideal if ideal > 0 else 0.0


def train(epochs=20, batch_size=256, lr=1e-3):
    user_ids, item_ids, item_content_map, pairs = asyncio.run(load_data())
    print(f"Loaded {len(user_ids)} users, {len(item_ids)} items, {len(pairs)} interactions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    num_users = max(user_ids) + 1
    num_items = max(item_ids) + 1

    dataset = InteractionDataset(pairs, item_ids, item_content_map)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = NeuralMF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=settings.embedding_dim,
        hidden_dims=settings.hidden_dims,
        dropout=settings.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = BPRLoss()

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            user_ids_b = batch["user_id"].to(device)
            pos_ids = batch["pos_item_id"].to(device)
            neg_ids = batch["neg_item_id"].to(device)
            content_pos = batch["content_pos"].to(device)
            content_neg = batch["content_neg"].to(device)

            pos_scores = model(user_ids_b, pos_ids, content_pos)
            neg_scores = model(user_ids_b, neg_ids, content_neg)

            loss = criterion(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "num_users": num_users,
                    "num_items": num_items,
                },
                settings.model_path,
            )
            print(f"  Checkpoint saved (loss: {best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {settings.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    asyncio.run(init_db())
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
