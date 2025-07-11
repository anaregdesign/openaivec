"""
```
from openaivec import task
```
"""

from dataclasses import dataclass
from typing import Type, TypeVar
from pydantic import BaseModel

__all__ = ['PreparedTask']

T = TypeVar('T', bound=BaseModel)



@dataclass(frozen=True)
class PreparedTask:
    """
    PreparedTask is a data class that represents a task to be executed.
    """
    instructions: str
    response_format: Type[T]
