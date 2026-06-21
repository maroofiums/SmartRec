from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import relationship
from app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    interactions = relationship("Interaction", back_populates="user")
    embedding_cache = relationship("UserEmbeddingCache", back_populates="user", uselist=False)


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    genres = Column(JSON, default=list)        # ["Action", "Thriller"]
    tags = Column(JSON, default=list)          # ["sequel", "blockbuster"]
    content_vector = Column(JSON, default=list) # pre-encoded content features
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    interactions = relationship("Interaction", back_populates="item")
    embedding_cache = relationship("ItemEmbeddingCache", back_populates="item", uselist=False)


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False, index=True)
    rating = Column(Float, nullable=True)      # explicit; None = implicit positive
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="interactions")
    item = relationship("Item", back_populates="interactions")


class UserEmbeddingCache(Base):
    """Pre-computed user embeddings to avoid re-inference on each request."""
    __tablename__ = "user_embedding_cache"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    embedding = Column(JSON, nullable=False)   # list[float]
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="embedding_cache")


class ItemEmbeddingCache(Base):
    """Pre-computed item embeddings (CF + content fused) for fast retrieval."""
    __tablename__ = "item_embedding_cache"

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, ForeignKey("items.id"), unique=True, nullable=False)
    embedding = Column(JSON, nullable=False)   # list[float]
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    item = relationship("Item", back_populates="embedding_cache")
