import sys
from typing import Any, Optional

from pydantic import BaseSettings, ValidationError, validator


class Settings(BaseSettings):
    """App Base setting

    See https://pydantic-docs.helpmanual.io/usage/settings/
    """

    title: str = "Speech2Text API"
    description: str = "Speech2Text API"

    api_v1_route: str = "/api/v1"
    openapi_route: str = "/openapi.json"

    host: str = "127.0.0.1"
    port: Any = 5000

    workers: int = 1

    debug_api: bool = False
    verbose: bool = False
    log_level_uvicorn: str = "info"

    sentry_dsn: Optional[str] = None

    statsd_host: Optional[str] = None
    statsd_prefix: str = "speech2text_api"

    class Config:
        env_file = ".env"
        env_prefix = "speech2text_api_"

    @validator("statsd_host")
    def statsd_host_valiadtor(cls: "Settings", v: Optional[str]) -> Optional[str]:
        if v is not None:
            assert ":" in v
        return v


try:
    settings = Settings()
except ValidationError as e:
    print(
        "Incorrect configuration:\n",
        e,
        "\nSee README for configuration details.",
        sep="\n",
        file=sys.stderr,
    )
    sys.exit(1)
