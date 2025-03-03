import json
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


class TestDeserialize(TestCase):
    def test_deserialize(self):
        cls = deserialize_base_model(Team.model_json_schema())
        json_schema = cls.model_json_schema()
        self.assertEqual("Team", json_schema["title"])
        self.assertEqual("object", json_schema["type"])
