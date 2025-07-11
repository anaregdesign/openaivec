from dataclasses import dataclass
from typing import Type, TypeVar
from pydantic import BaseModel

__all__ = ['PreparedTask']

T = TypeVar('T', bound=BaseModel)



@dataclass(frozen=True)
class PreparedTask:
    instructions: str
    response_format: Type[T]
