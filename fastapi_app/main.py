from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from fastapi_app.models import (
    CreateAlertRequest,
    CreateStrategyBacktestRequest,
    StrategyBacktestResponse,
)
from fastapi_app.services import StrategyBacktesterService

app = FastAPI(title="botMaker API", version="1.0.0")


def get_strategy_backtester_service() -> StrategyBacktesterService:
    return StrategyBacktesterService()


@app.get("/")
def root() -> str:
    return "Welcome to botMaker API"


@app.get("/api/alerts")
def get_alerts():
    return ["ping", "pong"]


@app.get("/api/alerts/{alert_id}")
def get_alert_by_id(alert_id: int):
    return {"id": alert_id, "status": "ok"}


@app.post("/api/alerts", status_code=201)
def create_alert(payload: CreateAlertRequest):
    return {"id": 1, **payload.model_dump()}


@app.get("/api/strategybacktester", response_model=StrategyBacktestResponse)

def get_strategy_backtester(
    service: StrategyBacktesterService = Depends(get_strategy_backtester_service),
):
    return service.dummy_return()


@app.get("/api/strategybacktester/{strategy_id}")
def get_strategy_backtester_by_id(strategy_id: int):
    return {"id": strategy_id, "status": "ok"}


@app.post("/api/strategybacktester", status_code=201)

def create_strategy_backtester(payload: CreateStrategyBacktestRequest):
    return {"id": 1, **payload.model_dump()}
