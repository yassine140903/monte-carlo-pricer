"""Application configuration, loaded from environment variables / .env."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "monte_carlo_pricer"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"

    # --- Data ---
    DEFAULT_LOOKBACK_YEARS: int = 3
    DATA_GRANULARITY: str = "1d"

    SUPPORTED_TICKERS: list[str] = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "BRK-B",
        "TSLA",
        "UNH",
        "JPM",
        "JNJ",
        "V",
        "XOM",
        "PG",
        "MA",
        "HD",
        "CVX",
        "MRK",
        "ABBV",
        "LLY",
    ]

    # --- MLflow ---
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


settings = Settings()
