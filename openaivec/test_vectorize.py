import os
from logging import basicConfig, Handler, StreamHandler
from typing import List
from unittest import TestCase

from openai import AzureOpenAI

from openaivec import VectorizedOpenAI

_h: Handler = StreamHandler()

basicConfig(handlers=[_h], level="DEBUG")


class TestVectorizedOpenAI(TestCase):
    def setUp(self):
        self.openai_client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        )

        self.model_name = "gpt-4o"

    def test_predict_str(self):
        system_message = """
        just repeat the user message
        """.strip()
        client = VectorizedOpenAI(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=system_message,
        )
        response: List[str] = client.predict(["hello", "world"])
        print(response)
