import os
from logging import basicConfig, Handler, StreamHandler
from typing import List
from unittest import TestCase

from openai import AzureOpenAI
from pydantic import BaseModel

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

        self.assertEqual(response, ["hello", "world"])

    def test_predict_structured(self):
        system_message = """
        return the color and taste of given fruit
        #example
        ## input
        apple

        ## output
        {{
            "name": "apple",
            "color": "red",
            "taste": "sweet"
        }}
        """

        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        client = VectorizedOpenAI(
            client=self.openai_client, model_name=self.model_name, system_message=system_message, response_format=Fruit
        )

        response: List[Fruit] = client.predict(["apple", "banana"])

        self.assertTrue(all(isinstance(item, Fruit) for item in response))
