from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.session import get_db
from app.models.orm import User, Item, Interaction
from app.schemas.schemas import (
    UserCreate, UserResponse,
    ItemCreate, ItemResponse,
    InteractionCreate, InteractionResponse,
    RecommendationResponse,
)
from app.services.recommender import recommender
from app.core.config import get_settings

settings = get_settings()
router = APIRouter()


# ── Users ───────────────────────────────────────────────────────────────────────

@router.post("/users/", response_model=UserResponse, status_code=201)
async def create_user(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.external_id == payload.external_id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="User already exists")
    user = User(external_id=payload.external_id)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/users/{external_id}", response_model=UserResponse)
async def get_user(external_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.external_id == external_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ── Items ────────────────────────────────────────────────────────────────────────

@router.post("/items/", response_model=ItemResponse, status_code=201)
async def create_item(payload: ItemCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(Item).where(Item.external_id == payload.external_id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Item already exists")
    item = Item(**payload.model_dump())
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


@router.get("/items/", response_model=list[ItemResponse])
async def list_items(
    skip: int = 0,
    limit: int = Query(default=50, le=200),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Item).offset(skip).limit(limit))
    return result.scalars().all()


# ── Interactions ─────────────────────────────────────────────────────────────────

@router.post("/interactions/", response_model=InteractionResponse, status_code=201)
async def log_interaction(payload: InteractionCreate, db: AsyncSession = Depends(get_db)):
    user_res = await db.execute(select(User).where(User.external_id == payload.user_external_id))
    user = user_res.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    item_res = await db.execute(select(Item).where(Item.external_id == payload.item_external_id))
    item = item_res.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    interaction = Interaction(user_id=user.id, item_id=item.id, rating=payload.rating)
    db.add(interaction)
    await db.commit()
    await db.refresh(interaction)
    return interaction


# ── Recommendations ───────────────────────────────────────────────────────────────

@router.get("/recommend/{user_external_id}", response_model=RecommendationResponse)
async def recommend(
    user_external_id: str,
    top_k: int = Query(default=None, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.external_id == user_external_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    k = top_k or settings.top_k
    recs = await recommender.get_recommendations(user=user, db=db, top_k=k)

    return RecommendationResponse(
        user_id=user.id,
        user_external_id=user.external_id,
        recommendations=recs,
        top_k=k,
    )


# ── Health ──────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    user_count = await db.execute(select(func.count(User.id)))
    item_count = await db.execute(select(func.count(Item.id)))
    return {
        "status": "ok",
        "model_loaded": recommender.model is not None,
        "users": user_count.scalar(),
        "items": item_count.scalar(),
    }
