from logging import Handler, StreamHandler, basicConfig
from typing import List
from unittest import TestCase

from openai import AsyncOpenAI
from pydantic import BaseModel

from openaivec.beta.responses import VectorizedResponsesOpenAI

_h: Handler = StreamHandler()

basicConfig(handlers=[_h], level="DEBUG")


class TestVectorizedResponsesOpenAI(TestCase):
    def setUp(self):
        self.openai_client = AsyncOpenAI()
        self.model_name = "gpt-4o-mini"

    def test_parse_str(self):
        system_message = """
        just repeat the user message
        """.strip()
        client = VectorizedResponsesOpenAI(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=system_message,
        )
        response: List[str] = client.parse(["apple", "orange", "banana", "pineapple"], batch_size=1)

        self.assertListEqual(response, ["apple", "orange", "banana", "pineapple"])

    def test_parse_structured(self):
        system_message = """
        return the color and taste of given fruit
        #example
        ## input
        apple

        ## output
        {
            "name": "apple",
            "color": "red",
            "taste": "sweet"
        }
        """

        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        client = VectorizedResponsesOpenAI(
            client=self.openai_client, model_name=self.model_name, system_message=system_message, response_format=Fruit
        )

        response: List[Fruit] = client.parse(["apple", "banana", "orange", "pinapple"], batch_size=1)

        self.assertTrue(all(isinstance(item, Fruit) for item in response))
