"""Необязательная FastAPI-обёртка над core service layer."""

import os
from pathlib import Path
from typing import Any

from .service import RoutingService


def create_app(model_path: str | Path | None = None) -> Any:
    """Создать приложение; FastAPI импортируется только при явном использовании."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as error:
        raise RuntimeError('Установите optional dependencies: pip install -e ".[api]"') from error

    resolved_path = model_path or os.environ.get("BANKING_MODEL_PATH")
    if not resolved_path:
        raise RuntimeError("Задайте BANKING_MODEL_PATH или передайте model_path")
    service = RoutingService.from_path(resolved_path)
    app = FastAPI(title="Banking ticket routing", version="0.1.0")

    class TicketRequest(BaseModel):
        text: str = Field(min_length=1)
        top_k: int = Field(default=3, ge=1, le=10)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/route")
    def route(request: TicketRequest) -> dict[str, Any]:
        try:
            return service.route(request.text, top_k=request.top_k)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
