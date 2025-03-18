from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from logging import Logger, getLogger
from typing import Generic, List, Type, TypeVar, cast

from openai import OpenAI, RateLimitError
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel

from openaivec.log import observe
from openaivec.util import backoff, map_unique_minibatch, map_unique_minibatch_parallel

__all__ = ["VectorizedLLM", "VectorizedOpenAI"]

_logger: Logger = getLogger(__name__)


def vectorize_system_message(system_message: str) -> str:
    return f"""
<SystemMessage>
    <ElementInstructions>
        <Instruction>{system_message}</Instruction>
    </ElementInstructions>
    <BatchInstructions>
        <Instruction>
            You will receive multiple user messages at once.
            Please provide an appropriate response to each message individually.
        </Instruction>
    </BatchInstructions>
    <Examples>
        <Example>
            <Input>
                {{
                    "user_messages": [
                        {{
                            "id": 1,
                            "body": "{{user_message_1}}"
                        }},
                        {{
                            "id": 2,
                            "body": "{{user_message_2}}"
                        }}
                    ]
                }}
            </Input>
            <Output>
                {{
                    "assistant_messages": [
                        {{
                            "id": 1,
                            "body": "{{assistant_response_1}}"
                        }},
                        {{
                            "id": 2,
                            "body": "{{assistant_response_2}}"
                        }}
                    ]
                }}
            </Output>
        </Example>
    </Examples>
</SystemMessage>
"""


T = TypeVar("T")


class Message(BaseModel, Generic[T]):
    id: int
    body: T


class Request(BaseModel):
    user_messages: List[Message[str]]


class Response(BaseModel, Generic[T]):
    assistant_messages: List[Message[T]]


class VectorizedLLM(Generic[T], metaclass=ABCMeta):
    @abstractmethod
    def predict(self, user_messages: List[str]) -> List[T]:
        pass

    @abstractmethod
    def predict_minibatch(self, user_messages: List[str], batch_size: int) -> List[T]:
        pass


@dataclass(frozen=True)
class VectorizedOpenAI(VectorizedLLM, Generic[T]):
    client: OpenAI
    model_name: str  # it would be the name of deployment for Azure
    system_message: str
    temperature: float = 0.0
    top_p: float = 1.0
    response_format: Type[T] = str
    is_parallel: bool = False
    _vectorized_system_message: str = field(init=False)
    _model_json_schema: dict = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_vectorized_system_message",
            vectorize_system_message(self.system_message),
        )

    @observe(_logger)
    @backoff(exception=RateLimitError, interval=60, max_retries=32)
    def request(self, user_messages: List[Message[str]]) -> ParsedChatCompletion[Response[T]]:
        response_format = self.response_format

        class MessageT(BaseModel):
            id: int
            body: response_format

        class ResponseT(BaseModel):
            assistant_messages: List[MessageT]

        completion: ParsedChatCompletion[ResponseT] = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._vectorized_system_message},
                {
                    "role": "user",
                    "content": Request(user_messages=user_messages).model_dump_json(),
                },
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            response_format=ResponseT,
        )
        return cast(ParsedChatCompletion[Response[T]], completion)

    @observe(_logger)
    def predict(self, user_messages: List[str]) -> List[T]:
        messages = [Message(id=i, body=message) for i, message in enumerate(user_messages)]
        completion = self.request(messages)
        response_dict = {
            message.id: message.body for message in completion.choices[0].message.parsed.assistant_messages
        }
        sorted_responses = [response_dict.get(m.id, None) for m in messages]
        return sorted_responses

    @observe(_logger)
    def predict_minibatch(self, user_messages: List[str], batch_size: int) -> List[T]:
        if self.is_parallel:
            return map_unique_minibatch_parallel(user_messages, batch_size, self.predict)
        else:
            return map_unique_minibatch(user_messages, batch_size, self.predict)
