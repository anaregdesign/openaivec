import asyncio
from dataclasses import dataclass, field
from logging import getLogger
from typing import Generic, List, TypeVar, Type, cast
from openai import AsyncOpenAI, RateLimitError
from openai.types.responses import ParsedResponse
from pydantic import BaseModel

from openaivec.log import observe
from openaivec.responses import Message, Request, Response, VectorizedResponses, _vectorize_system_message
from openaivec.util import backoff, map_unique_minibatch_async

__all__ = ["VectorizedResponsesOpenAI"]

T = TypeVar("T")
_LOGGER = getLogger(__name__)


@dataclass(frozen=True)
class VectorizedResponsesOpenAI(VectorizedResponses, Generic[T]):
    client: AsyncOpenAI
    model_name: str  # it would be the name of deployment for Azure
    system_message: str
    temperature: float = 0.0
    top_p: float = 1.0
    response_format: Type[T] = str
    _vectorized_system_message: str = field(init=False)
    _model_json_schema: dict = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_vectorized_system_message",
            _vectorize_system_message(self.system_message),
        )

    @observe(_LOGGER)
    @backoff(exception=RateLimitError, scale=60, max_retries=16)
    async def _request_llm(self, user_messages: List[Message[str]]) -> ParsedResponse[Response[T]]:
        response_format = self.response_format

        class MessageT(BaseModel):
            id: int
            body: response_format  # type: ignore

        class ResponseT(BaseModel):
            assistant_messages: List[MessageT]

        # Directly await the async call instead of using asyncio.run()
        completion: ParsedResponse[ResponseT] = await self.client.responses.parse(
            model=self.model_name,
            instructions=self._vectorized_system_message,
            input=Request(user_messages=user_messages).model_dump_json(),
            temperature=self.temperature,
            top_p=self.top_p,
            text_format=ResponseT,
        )
        return cast(ParsedResponse[Response[T]], completion)

    @observe(_LOGGER)
    async def _predict_chunk(self, user_messages: List[str]) -> List[T]:
        messages = [Message(id=i, body=message) for i, message in enumerate(user_messages)]
        responses: ParsedResponse[Response[T]] = await self._request_llm(messages)
        response_dict = {message.id: message.body for message in responses.output_parsed.assistant_messages}
        sorted_responses = [response_dict.get(m.id, None) for m in messages]
        return sorted_responses

    @observe(_LOGGER)
    def parse(self, inputs: List[str], batch_size: int) -> List[T]:
        return asyncio.run(
            map_unique_minibatch_async(
                inputs,
                batch_size,
                self._predict_chunk,
            )
        )
