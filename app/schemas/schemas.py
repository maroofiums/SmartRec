from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Users ──────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    external_id: str = Field(..., description="Unique external identifier for the user")


class UserResponse(BaseModel):
    id: int
    external_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Items ──────────────────────────────────────────────────────────────────────

class ItemCreate(BaseModel):
    external_id: str
    title: str
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    content_vector: list[float] = Field(
        default_factory=list,
        description="Pre-encoded content feature vector (e.g. from TF-IDF or sentence embeddings)"
    )


class ItemResponse(BaseModel):
    id: int
    external_id: str
    title: str
    genres: list[str]
    tags: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Interactions ───────────────────────────────────────────────────────────────

class InteractionCreate(BaseModel):
    user_external_id: str
    item_external_id: str
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Explicit rating; omit for implicit feedback")


class InteractionResponse(BaseModel):
    id: int
    user_id: int
    item_id: int
    rating: Optional[float]
    timestamp: datetime

    model_config = {"from_attributes": True}


# ── Recommendations ────────────────────────────────────────────────────────────

class RecommendedItem(BaseModel):
    item_id: int
    external_id: str
    title: str
    genres: list[str]
    score: float = Field(..., description="Predicted relevance score in [0, 1]")


class RecommendationResponse(BaseModel):
    user_id: int
    user_external_id: str
    recommendations: list[RecommendedItem]
    model: str = "neural-mf-hybrid"
    top_k: int
