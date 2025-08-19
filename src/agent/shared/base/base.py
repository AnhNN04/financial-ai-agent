# src/stock_assistant/shared/base/entity.py
from abc import ABC
from pydantic import BaseModel
from datetime import datetime


class BaseEntity(BaseModel, ABC):
    """Base entity class for domain objects"""
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    
    class ConfigDict:
        arbitrary_types_allowed = True
        validate_assignment = True