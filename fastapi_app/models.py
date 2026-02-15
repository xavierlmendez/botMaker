from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class CreateAlertRequest(BaseModel):
    name: str = Field(..., example="Price Alert")
    threshold: int = Field(..., example=10)


class CreateStrategyBacktestRequest(BaseModel):
    name: str = Field(..., example="Moving Average Crossover")
    threshold: int = Field(..., example=10)


class TickerAllocation(BaseModel):
    symbol: str
    displayName: str
    allocation: float


class Timeframe(BaseModel):
    start: str
    end: str


class StrategySettings(BaseModel):
    initialCapital: int
    rebalanceFrequency: str
    benchmark: str


class StrategyIndicator(BaseModel):
    name: str
    parameters: Dict[str, Any]


class RiskManagement(BaseModel):
    stopLoss: float
    takeProfit: float
    positionSizing: str


class StrategyAdditionalProperties(BaseModel):
    indicators: List[StrategyIndicator]
    riskManagement: RiskManagement
    notes: str


class StrategyBacktestResponse(BaseModel):
    strategyId: str
    strategyName: str
    description: str
    tickers: List[TickerAllocation]
    timeframe: Timeframe
    settings: StrategySettings
    additionalProperties: StrategyAdditionalProperties
