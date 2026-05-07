from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Agro-Vision"
    api_prefix: str = "/api/v1"
    debug: bool = False

    jwt_secret: str = Field(default="dev-secret-change-me")
    jwt_alg: str = "HS256"
    jwt_ttl_minutes: int = 60 * 24 * 7

    cookie_name: str = "agrovision_session"
    cookie_secure: bool = False
    cookie_samesite: str = "lax"

    database_url: str = f"sqlite:///{PROJECT_ROOT / 'agrovision.db'}"
    turso_auth_token: str | None = None

    upload_dir: Path = PROJECT_ROOT / "uploads"
    maps_dir: Path = PROJECT_ROOT / "app" / "static" / "maps"
    static_dir: Path = PROJECT_ROOT / "app" / "static"
    model_path: Path = PROJECT_ROOT / "ml" / "weights" / "cnn_lstm.pth"
    labels_csv: Path = PROJECT_ROOT / "metadata" / "plot_metadata.csv"

    max_upload_mb: int = 200
    num_timesteps: int = 12

    health_ai_provider: str = "heuristic"
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    health_ai_timeout_s: float = 8.0

    cors_origins: list[str] = []


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    s.maps_dir.mkdir(parents=True, exist_ok=True)
    return s
