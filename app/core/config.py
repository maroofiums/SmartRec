from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/smartrec"

    # Model
    embedding_dim: int = 64
    hidden_dims: list[int] = [128, 64, 32]
    dropout: float = 0.2
    model_path: str = "smartrec_model.pt"

    # Recommendation
    top_k: int = 10
    cache_embeddings: bool = True

    # API
    app_name: str = "SmartRec"
    debug: bool = False

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
