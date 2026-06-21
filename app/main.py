from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import select, func

from app.api.routes.recommend import router
from app.core.config import get_settings
from app.db.session import init_db, AsyncSessionLocal
from app.models.orm import User, Item
from app.services.recommender import recommender

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()

    # Load model if checkpoint exists
    async with AsyncSessionLocal() as db:
        user_count = (await db.execute(select(func.count(User.id)))).scalar()
        item_count = (await db.execute(select(func.count(Item.id)))).scalar()

    try:
        recommender.load_model(num_users=user_count + 1, num_items=item_count + 1)
        print(f"Model loaded — {user_count} users, {item_count} items")
    except FileNotFoundError:
        print("No model checkpoint found. Train the model first with: python scripts/train.py")

    yield
    # Shutdown (nothing to clean up)


app = FastAPI(
    title=settings.app_name,
    description="Hybrid Neural Matrix Factorization Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, tags=["SmartRec"])
