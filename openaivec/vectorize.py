import json
from dataclasses import dataclass
from typing import List

from openai import OpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel

from openaivec.util import map_with_minibatch

__ALL__ = ["BatchOpenAI"]


def vectorize_system_message(system_message: str) -> str:
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

def gather_user_message(user_messages: List[str]) -> str:
    return json.dumps({
        "user_messages": [
            Message(id=i, text=message)
            for i, message in enumerate(user_messages)
        ]
    }, ensure_ascii=False)

class Message(BaseModel):
    id: int
    text: str


class Request(BaseModel):
    user_messages: List[Message]

class Response(BaseModel):
    assistant_messages: List[Message]


@dataclass(frozen=True)
class VectorizedOpenAI:
    client: OpenAI
    model_name: str ## it would be the name of deployment for Azure
    system_message: str
    temperature: float = 0.0
    top_p: float = 1.0

    def request(self, user_messages: List[Message]) -> ParsedChatCompletion[Response]:
        system_message = vectorize_system_message(self.system_message)
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": Request(user_messages=user_messages).model_dump_json()}
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            response_format=Response
        )
        return completion

    def predict(self, user_messages: List[str]) -> List[str]:
        messages = [Message(id=i, text=message) for i, message in enumerate(user_messages)]
        completion = self.request(m)
        response_dict = {
            message.id: message.text
            for message in completion.choices[0].message.parsed.assistant_messages
        }
        sorted_responses = [response_dict[m.id] for m in messages]
        return sorted_responses

    def predict_minibatch(self, user_messages: List[str], batch_size: int) -> List[str]:
        return map_with_minibatch(user_messages, batch_size, self.predict)
