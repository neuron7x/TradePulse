# SPDX-License-Identifier: MIT
"""Runtime validation with Pydantic for TradePulse data models.

This module provides Pydantic models for runtime validation of all
public data structures, ensuring type safety and data integrity at runtime.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

try:
    from pydantic import BaseModel, Field, validator, ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore


if PYDANTIC_AVAILABLE:
    class TickerModel(BaseModel):
        """Validated market data tick."""
        
        model_config = ConfigDict(frozen=True, strict=True)
        
        ts: float = Field(..., description="Unix timestamp")
        price: float = Field(..., gt=0, description="Price (must be positive)")
        volume: float = Field(default=0.0, ge=0, description="Volume (non-negative)")
        
        @validator('ts')
        def validate_timestamp(cls, v: float) -> float:
            """Validate timestamp is reasonable."""
            if v < 0:
                raise ValueError("Timestamp cannot be negative")
            if v > 9999999999:  # Year ~2286
                raise ValueError("Timestamp is too far in the future")
            return v
        
        @validator('price')
        def validate_price(cls, v: float) -> float:
            """Validate price is reasonable."""
            if not (-1e15 < v < 1e15):
                raise ValueError("Price value is out of reasonable range")
            return v


    class FeatureResultModel(BaseModel):
        """Validated feature/indicator result."""
        
        model_config = ConfigDict(strict=False)  # Allow flexible metadata
        
        name: str = Field(..., min_length=1, description="Feature name")
        value: Any = Field(..., description="Feature value")
        metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
        
        @validator('name')
        def validate_name(cls, v: str) -> str:
            """Validate feature name is reasonable."""
            if len(v) > 100:
                raise ValueError("Feature name is too long (max 100 chars)")
            return v


    class BacktestResultModel(BaseModel):
        """Validated backtest result."""
        
        model_config = ConfigDict(frozen=True)
        
        pnl: float = Field(..., description="Profit and loss")
        max_dd: float = Field(..., le=0, description="Maximum drawdown (non-positive)")
        trades: int = Field(..., ge=0, description="Number of trades (non-negative)")
        
        @validator('max_dd')
        def validate_max_dd(cls, v: float) -> float:
            """Validate max drawdown is non-positive."""
            if v > 0:
                raise ValueError("Maximum drawdown must be non-positive")
            return v
        
        @validator('trades')
        def validate_trades(cls, v: int) -> int:
            """Validate number of trades is reasonable."""
            if v > 1000000:
                raise ValueError("Number of trades is unreasonably high")
            return v


    class OrderModel(BaseModel):
        """Validated order specification."""
        
        model_config = ConfigDict(frozen=True, strict=True)
        
        side: str = Field(..., pattern="^(buy|sell)$", description="Order side")
        quantity: float = Field(..., gt=0, description="Order quantity (positive)")
        price: Optional[float] = Field(None, gt=0, description="Limit price (if applicable)")
        order_type: str = Field(
            default="market",
            pattern="^(market|limit)$",
            description="Order type"
        )
        
        @validator('quantity')
        def validate_quantity(cls, v: float) -> float:
            """Validate order quantity is reasonable."""
            if v > 1e10:
                raise ValueError("Order quantity is unreasonably high")
            return v


    class StrategyConfigModel(BaseModel):
        """Validated strategy configuration."""
        
        model_config = ConfigDict(strict=False)  # Allow flexible params
        
        name: str = Field(..., min_length=1, max_length=100, description="Strategy name")
        params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
        enabled: bool = Field(default=True, description="Whether strategy is enabled")
        
        @validator('params')
        def validate_params(cls, v: Dict[str, Any]) -> Dict[str, Any]:
            """Validate strategy parameters."""
            if len(v) > 100:
                raise ValueError("Too many strategy parameters (max 100)")
            return v


def validate_ticker(data: Dict[str, Any]) -> TickerModel:
    """Validate ticker data with Pydantic.
    
    Args:
        data: Dictionary with ticker data
        
    Returns:
        Validated TickerModel
        
    Raises:
        ValidationError: If data is invalid
        RuntimeError: If Pydantic is not available
    """
    if not PYDANTIC_AVAILABLE:
        raise RuntimeError("Pydantic is not installed")
    return TickerModel(**data)


def validate_feature_result(data: Dict[str, Any]) -> FeatureResultModel:
    """Validate feature result with Pydantic.
    
    Args:
        data: Dictionary with feature result data
        
    Returns:
        Validated FeatureResultModel
        
    Raises:
        ValidationError: If data is invalid
        RuntimeError: If Pydantic is not available
    """
    if not PYDANTIC_AVAILABLE:
        raise RuntimeError("Pydantic is not installed")
    return FeatureResultModel(**data)


def validate_backtest_result(data: Dict[str, Any]) -> BacktestResultModel:
    """Validate backtest result with Pydantic.
    
    Args:
        data: Dictionary with backtest result data
        
    Returns:
        Validated BacktestResultModel
        
    Raises:
        ValidationError: If data is invalid
        RuntimeError: If Pydantic is not available
    """
    if not PYDANTIC_AVAILABLE:
        raise RuntimeError("Pydantic is not installed")
    return BacktestResultModel(**data)


def validate_order(data: Dict[str, Any]) -> OrderModel:
    """Validate order with Pydantic.
    
    Args:
        data: Dictionary with order data
        
    Returns:
        Validated OrderModel
        
    Raises:
        ValidationError: If data is invalid
        RuntimeError: If Pydantic is not available
    """
    if not PYDANTIC_AVAILABLE:
        raise RuntimeError("Pydantic is not installed")
    return OrderModel(**data)


def validate_strategy_config(data: Dict[str, Any]) -> StrategyConfigModel:
    """Validate strategy configuration with Pydantic.
    
    Args:
        data: Dictionary with strategy config data
        
    Returns:
        Validated StrategyConfigModel
        
    Raises:
        ValidationError: If data is invalid
        RuntimeError: If Pydantic is not available
    """
    if not PYDANTIC_AVAILABLE:
        raise RuntimeError("Pydantic is not installed")
    return StrategyConfigModel(**data)


__all__ = [
    "PYDANTIC_AVAILABLE",
    "TickerModel",
    "FeatureResultModel",
    "BacktestResultModel",
    "OrderModel",
    "StrategyConfigModel",
    "validate_ticker",
    "validate_feature_result",
    "validate_backtest_result",
    "validate_order",
    "validate_strategy_config",
]
