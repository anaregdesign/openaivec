"""
`openaivec.task` is a module that provides predifined tasks for the OpenAI API.
`PreparedTask` is a data class that represents a task to be executed, including instructions, response format, temperature, and top_p settings.

```
from openai import OpenAI
from openaivec import task
from openaivec.responses import BatchResponses

translation_task: BatchResponses = BatchResponses.of_task(
    client=OpenAI(),
    model_name="gpt-4.1-mini",
    task=task.MULTILINGUAL_TRANSLATION_TASK
)
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
    temperature: float = 0.0
    top_p: float = 1.0
