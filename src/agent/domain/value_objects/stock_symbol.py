from pydantic import BaseModel, field_validator, ConfigDict
from typing import Literal

MarketType = Literal["HOSE", "HNX", "UPCoM"]

class StockSymbol(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)  # <- key để hashable

    symbol: str
    market: MarketType

    @field_validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) < 2 or len(v) > 10:
            raise ValueError('Stock symbol must be between 2-10 characters')
        return v.upper().strip()

    def __str__(self) -> str:
        return f"{self.symbol}:{self.market}"
