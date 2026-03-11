"""
Application configuration — loads from .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # JWT
    SECRET_KEY: str = "change-me-in-production-must-be-at-least-32-chars!!"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # DeepFace
    DEEPFACE_MODEL: str = "ArcFace"
    DEEPFACE_DETECTOR: str = "opencv"          # lighter detector; swap for retinaface in prod
    DEEPFACE_DISTANCE_METRIC: str = "cosine"
    VERIFICATION_THRESHOLD: float = 0.40
    LIVENESS_THRESHOLD: float = 0.80

    # Storage
    ENROLLED_FACES_DIR: str = "./data/enrolled"
    UPLOAD_DIR: str = "./data/uploads"

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:8000,http://127.0.0.1:8000"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    def ensure_dirs(self) -> None:
        Path(self.ENROLLED_FACES_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
