"""VectorizeOpenAI Module.

This module provides an interface for the OpenAI API using vectorized system messages.
It defines classes to represent messages, requests, and responses, along with a
VectorizedOpenAI class that handles API requests and response parsing.

Example:
    client = OpenAI(...)
    vectorized_api = VectorizedOpenAI(
        client=client,
        model_name="your_deployment_name",
        system_message="Your detailed system instructions."
    )
    responses = vectorized_api.predict(["Hello", "How are you?"])
"""

from dataclasses import dataclass, field
from logging import Logger, getLogger
from typing import List

from openai import OpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel

from openaivec.log import observe
from openaivec.util import map_unique_minibatch_parallel

__ALL__ = ["VectorizedOpenAI"]

_logger: Logger = getLogger(__name__)


def vectorize_system_message(system_message: str) -> str:
    """Converts a system message into a predefined XML format.

    Args:
        system_message (str): The system message to be vectorized.

    Returns:
        str: The vectorized system message in XML format.
    """
    return f"""
<SystemMessage>
    <Instructions>
        <Instruction>{system_message}</Instruction>
        <Instruction>
            You will receive multiple user messages at once.
            Please provide an appropriate response to each message individually.
        </Instruction>
    </Instructions>
    <Examples>
        <Example>
            <Input>
                {{
                    "user_messages": [
                        {{
                            "id": 1,
                            "text": "{{user_message_1}}"
                        }},
                        {{
                            "id": 2,
                            "text": "{{user_message_2}}"
                        }}
                    ]
                }}
            </Input>
            <Output>
                {{
                    "assistant_messages": [
                        {{
                            "id": 1,
                            "text": "{{assistant_response_1}}"
                        }},
                        {{
                            "id": 2,
                            "text": "{{assistant_response_2}}"
                        }}
                    ]
                }}
            </Output>
        </Example>
    </Examples>
</SystemMessage>
"""


class Message(BaseModel):
    """Represents a message with an ID and text.

    Attributes:
        id (int): The ID of the message.
        text (str): The text content of the message.
    """

    id: int
    text: str


class Request(BaseModel):
    """Represents a request containing user messages.

    Attributes:
        user_messages (List[Message]): A list of user messages.
    """

    user_messages: List[Message]


class Response(BaseModel):
    """Represents a response containing assistant messages.

    Attributes:
        assistant_messages (List[Message]): A list of assistant messages.
    """

    assistant_messages: List[Message]


@dataclass(frozen=True)
class VectorizedOpenAI:
    """A class to interact with the OpenAI API using vectorized system messages.

    Attributes:
        client (OpenAI): The OpenAI client.
        model_name (str): The name of the model or deployment.
        system_message (str): The system message to be vectorized.
        temperature (float): The temperature setting for the model.
        top_p (float): The top_p setting for the model.
        _vectorized_system_message (str): The vectorized system message.
    """

    client: OpenAI
    model_name: str  # it would be the name of deployment for Azure
    system_message: str
    temperature: float = 0.0
    top_p: float = 1.0
    _vectorized_system_message: str = field(init=False)

    def __post_init__(self):
        """Performs post-initialization to set the vectorized system message."""
        object.__setattr__(
            self,
            "_vectorized_system_message",
            vectorize_system_message(self.system_message),
        )

    @observe(_logger)
    def request(self, user_messages: List[Message]) -> ParsedChatCompletion[Response]:
        """Sends a request to the OpenAI API with user messages.

        Args:
            user_messages (List[Message]): A list of user messages.

        Returns:
            ParsedChatCompletion[Response]: The parsed response from the API.
        """
        completion = self.client.beta.chat.completions.parse(
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
            response_format=Response,
        )
        return completion

    @observe(_logger)
    def predict(self, user_messages: List[str]) -> List[str]:
        """Predicts responses for a list of user messages.

        Args:
            user_messages (List[str]): A list of user message texts.

        Returns:
            List[str]: A list of predicted assistant responses.
        """
        messages = [Message(id=i, text=message) for i, message in enumerate(user_messages)]
        completion = self.request(messages)
        response_dict = {
            message.id: message.text for message in completion.choices[0].message.parsed.assistant_messages
        }
        sorted_responses = [response_dict.get(m.id, None) for m in messages]
        return sorted_responses

    @observe(_logger)
    def predict_minibatch(self, user_messages: List[str], batch_size: int) -> List[str]:
        """Predicts responses for a list of user messages in minibatches.

        Args:
            user_messages (List[str]): A list of user message texts.
            batch_size (int): The size of each minibatch.

        Returns:
            List[str]: A list of predicted assistant responses.
        """
        return map_unique_minibatch_parallel(user_messages, batch_size, self.predict)
