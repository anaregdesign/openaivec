import unittest
import asyncio
from logging import Handler, StreamHandler, basicConfig
from typing import List

from openai import AsyncOpenAI
from pydantic import BaseModel

from openaivec.aio import AsyncBatchResponses

_h: Handler = StreamHandler()

basicConfig(handlers=[_h], level="DEBUG")


class TestAsyncBatchResponses(unittest.TestCase):
    def setUp(self):
        self.openai_client = AsyncOpenAI()
        self.model_name = "gpt-4.1-nano"

    def test_parse_str(self):
        system_message = """
        just repeat the user message
        """.strip()
        client = AsyncBatchResponses(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=system_message,
        )
        response: List[str] = asyncio.run(client.parse(["apple", "orange", "banana", "pineapple"], batch_size=1))
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
        input_fruits = ["apple", "banana", "orange", "pineapple"]

        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        client = AsyncBatchResponses(
            client=self.openai_client, model_name=self.model_name, system_message=system_message, response_format=Fruit
        )
        response: List[Fruit] = asyncio.run(client.parse(input_fruits, batch_size=1))
        self.assertEqual(len(response), len(input_fruits))
        for i, item in enumerate(response):
            self.assertIsInstance(item, Fruit)
            self.assertEqual(item.name.lower(), input_fruits[i].lower())
            self.assertIsInstance(item.color, str)
            self.assertTrue(len(item.color) > 0)
            self.assertIsInstance(item.taste, str)
            self.assertTrue(len(item.taste) > 0)

    def test_parse_structured_empty_input(self):
        system_message = """
        return the color and taste of given fruit
        """

        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        client = AsyncBatchResponses(
            client=self.openai_client, model_name=self.model_name, system_message=system_message, response_format=Fruit
        )
        response: List[Fruit] = asyncio.run(client.parse([], batch_size=1))
        self.assertListEqual(response, [])

    def test_parse_structured_batch_size(self):
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
        input_fruits = ["apple", "banana", "orange", "pineapple"]

        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        client = AsyncBatchResponses(
            client=self.openai_client, model_name=self.model_name, system_message=system_message, response_format=Fruit
        )
        response_bs2: List[Fruit] = asyncio.run(client.parse(input_fruits, batch_size=2))
        self.assertEqual(len(response_bs2), len(input_fruits))
        for i, item in enumerate(response_bs2):
            self.assertIsInstance(item, Fruit)
            self.assertEqual(item.name.lower(), input_fruits[i].lower())
            self.assertIsInstance(item.color, str)
            self.assertTrue(len(item.color) > 0)
            self.assertIsInstance(item.taste, str)
            self.assertTrue(len(item.taste) > 0)

        response_bs4: List[Fruit] = asyncio.run(client.parse(input_fruits, batch_size=4))
        self.assertEqual(len(response_bs4), len(input_fruits))
        for i, item in enumerate(response_bs4):
            self.assertIsInstance(item, Fruit)
            self.assertEqual(item.name.lower(), input_fruits[i].lower())
            self.assertIsInstance(item.color, str)
            self.assertTrue(len(item.color) > 0)
            self.assertIsInstance(item.taste, str)
            self.assertTrue(len(item.taste) > 0)
