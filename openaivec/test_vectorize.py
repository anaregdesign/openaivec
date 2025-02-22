import os
from logging import basicConfig, Handler, StreamHandler
from types import SimpleNamespace
from typing import List
from unittest import TestCase
from unittest.mock import MagicMock

from openai import AzureOpenAI
from openai.types.chat import ParsedChatCompletion

from openaivec import VectorizedOpenAI
from openaivec.vectorize import Message, Response

_h: Handler = StreamHandler()

basicConfig(handlers=[_h], level="DEBUG")


def create_dummy_parse(messages: List[Message]) -> ParsedChatCompletion[Response]:
    response = Response(
        assistant_messages=[Message(id=i, text=f"response_of_{m.text}") for i, m in enumerate(messages)]
    )
    dummy_message = SimpleNamespace(parsed=response)
    dummy_choice = SimpleNamespace(message=dummy_message)
    dummy_completion = SimpleNamespace(choices=[dummy_choice])

    return dummy_completion


class TestVectorizedOpenAI(TestCase):
    def setUp(self):
        self.openai_client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        )
        self.system_message = """
        just repeat the user message
        """
        self.model_name = "gpt-4o"

        self.client = VectorizedOpenAI(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=self.system_message,
        )

    def test_predict(self):
        response: List[str] = self.client.predict(["hello", "world"])
        print(response)
