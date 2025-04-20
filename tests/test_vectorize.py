from logging import Handler, StreamHandler, basicConfig
from typing import List
from unittest import TestCase

from openai import OpenAI
from pydantic import BaseModel

from openaivec import VectorizedOpenAI

_h: Handler = StreamHandler()

basicConfig(handlers=[_h], level="DEBUG")


class TestVectorizedOpenAI(TestCase):
    def setUp(self):
        self.openai_client = OpenAI()
        self.model_name = "gpt-4o-mini"

    def test_predict_str(self):
        system_message = """
        just repeat the user message
        """.strip()
        client = VectorizedOpenAI(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=system_message,
        )
        response: List[str] = client._predict_chunk(["hello", "world"])

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

        response: List[Fruit] = client._predict_chunk(["apple", "banana"])

        self.assertTrue(all(isinstance(item, Fruit) for item in response))
