import json
from dataclasses import dataclass
from typing import List

from openai import OpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel


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
            {"id": i, "text": message}
            for i, message in enumerate(user_messages)
        ]
    }, ensure_ascii=False)

class Message(BaseModel):
    id: int
    text: str

class Response(BaseModel):
    assistant_messages: List[Message]


@dataclass(frozen=True)
class VectorizedOpenAI:
    client: OpenAI
    model_name: str ## it would be the name of deployment for Azure
    system_message: str

    def request(self, user_messages: List[str]) -> ParsedChatCompletion[Response]:
        system_message = vectorize_system_message(self.system_message)
        user_message = gather_user_message(user_messages)
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format=Response
        )
        return completion

    def predict(self, user_messages: List[str]) -> List[str]:
        completion = self.request(user_messages)
        return [message.text for message in completion.response.assistant_messages]




