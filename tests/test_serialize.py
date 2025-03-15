from enum import Enum
from typing import List
from unittest import TestCase

from pydantic import BaseModel

from openaivec.serialize import deserialize_base_model


class Gender(str, Enum):
    FEMALE = "FEMALE"
    MALE = "MALE"


class Person(BaseModel):
    name: str
    age: int
    gender: Gender


class Team(BaseModel):
    name: str
    members: List[Person]
    rules: List[str]


class Matrix(BaseModel):
    data: List[List[float]]


class TestDeserialize(TestCase):
    def test_deserialize(self):
        cls = deserialize_base_model(Team.model_json_schema())
        json_schema = cls.model_json_schema()
        self.assertEqual("Team", json_schema["title"])
        self.assertEqual("object", json_schema["type"])
        self.assertEqual("string", json_schema["properties"]["name"]["type"])
        self.assertEqual("array", json_schema["properties"]["members"]["type"])
        self.assertEqual("array", json_schema["properties"]["rules"]["type"])
        self.assertEqual("string", json_schema["properties"]["rules"]["items"]["type"])

    def test_deserialize_with_nested_list(self):
        cls = deserialize_base_model(Matrix.model_json_schema())
        json_schema = cls.model_json_schema()
        self.assertEqual("Matrix", json_schema["title"])
        self.assertEqual("object", json_schema["type"])
        self.assertEqual("array", json_schema["properties"]["data"]["type"])
        self.assertEqual("array", json_schema["properties"]["data"]["items"]["type"])
        self.assertEqual("number", json_schema["properties"]["data"]["items"]["items"]["type"])
