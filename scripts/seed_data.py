"""
Seed the database with synthetic users, items, and interaction data.

Usage:
    python scripts/seed_data.py
"""

import asyncio
import random
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.db.session import AsyncSessionLocal, init_db
from app.models.orm import User, Item, Interaction

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Romance", "Horror", "Animation"]
TAGS = ["sequel", "blockbuster", "indie", "critically-acclaimed", "award-winner", "cult-classic"]

MOVIES = [
    ("The Dark Knight", ["Action", "Thriller"]),
    ("Inception", ["Sci-Fi", "Thriller"]),
    ("Interstellar", ["Sci-Fi", "Drama"]),
    ("Parasite", ["Drama", "Thriller"]),
    ("The Grand Budapest Hotel", ["Comedy", "Drama"]),
    ("Mad Max: Fury Road", ["Action", "Sci-Fi"]),
    ("La La Land", ["Romance", "Drama"]),
    ("Get Out", ["Horror", "Thriller"]),
    ("Spider-Man: Into the Spider-Verse", ["Animation", "Action"]),
    ("Arrival", ["Sci-Fi", "Drama"]),
    ("Knives Out", ["Comedy", "Thriller"]),
    ("Everything Everywhere All at Once", ["Sci-Fi", "Comedy"]),
    ("The Shawshank Redemption", ["Drama"]),
    ("Pulp Fiction", ["Crime", "Drama"]),
    ("Spirited Away", ["Animation", "Fantasy"]),
    ("The Social Network", ["Drama", "Thriller"]),
    ("Heat", ["Action", "Crime"]),
    ("No Country for Old Men", ["Thriller", "Drama"]),
    ("Blade Runner 2049", ["Sci-Fi", "Thriller"]),
    ("Whiplash", ["Drama", "Music"]),
]


def make_content_vector(genres: list[str], dim: int = 32) -> list[float]:
    """Simple genre one-hot + noise as a stand-in for real content embeddings."""
    genre_map = {g: i for i, g in enumerate(GENRES)}
    vec = np.zeros(dim, dtype=np.float32)
    for g in genres:
        if g in genre_map:
            vec[genre_map[g]] = 1.0
    vec[len(GENRES):] = np.random.randn(dim - len(GENRES)).astype(np.float32) * 0.1
    return vec.tolist()


async def seed():
    await init_db()

    async with AsyncSessionLocal() as db:
        # Create users
        users = []
        for i in range(1, 201):
            user = User(external_id=f"user_{i:04d}")
            db.add(user)
            users.append(user)
        await db.flush()

        # Create items
        items = []
        for idx, (title, genres) in enumerate(MOVIES):
            item = Item(
                external_id=f"movie_{idx+1:04d}",
                title=title,
                genres=genres,
                tags=random.sample(TAGS, k=random.randint(1, 3)),
                content_vector=make_content_vector(genres),
            )
            db.add(item)
            items.append(item)
        await db.flush()

        # Create interactions — each user rates 5–12 random movies
        for user in users:
            n_interactions = random.randint(5, 12)
            sampled_items = random.sample(items, k=n_interactions)
            for item in sampled_items:
                rating = round(random.uniform(2.5, 5.0), 1)
                db.add(Interaction(user_id=user.id, item_id=item.id, rating=rating))

        await db.commit()

    print(f"Seeded {len(users)} users, {len(items)} items")
    print("Interactions created (~5–12 per user)")


if __name__ == "__main__":
    asyncio.run(seed())
