from enum import Enum
from typing import List
from unittest import TestCase

from pydantic import BaseModel, Field

from openaivec.serialize import deserialize_base_model, serialize_base_model


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


class ModelWithDescriptions(BaseModel):
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score for sentiment (0.0-1.0)")


class ModelWithoutDescriptions(BaseModel):
    name: str
    age: int


class MixedModel(BaseModel):
    name: str  # No description
    age: int = Field()  # Field() without description
    email: str = Field(description="User's email address")  # With description


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

    def test_field_descriptions_preserved(self):
        """Test that Field descriptions are preserved during serialization/deserialization."""
        original = ModelWithDescriptions
        original_schema = original.model_json_schema()
        
        # Serialize and deserialize
        serialized = serialize_base_model(original)
        deserialized = deserialize_base_model(serialized)
        deserialized_schema = deserialized.model_json_schema()
        
        # Check that descriptions are preserved
        original_sentiment_desc = original_schema['properties']['sentiment'].get('description')
        deserialized_sentiment_desc = deserialized_schema['properties']['sentiment'].get('description')
        self.assertEqual(original_sentiment_desc, deserialized_sentiment_desc)
        
        original_confidence_desc = original_schema['properties']['confidence'].get('description')
        deserialized_confidence_desc = deserialized_schema['properties']['confidence'].get('description')
        self.assertEqual(original_confidence_desc, deserialized_confidence_desc)
        
        # Test that instances can be created
        instance = deserialized(sentiment="positive", confidence=0.95)
        self.assertEqual(instance.sentiment, "positive")
        self.assertEqual(instance.confidence, 0.95)

    def test_model_without_descriptions(self):
        """Test that models without Field descriptions work correctly."""
        original = ModelWithoutDescriptions
        
        # Serialize and deserialize
        serialized = serialize_base_model(original)
        deserialized = deserialize_base_model(serialized)
        deserialized_schema = deserialized.model_json_schema()
        
        # Check that no descriptions are present (should be None or absent)
        self.assertIsNone(deserialized_schema['properties']['name'].get('description'))
        self.assertIsNone(deserialized_schema['properties']['age'].get('description'))
        
        # Test that instances can be created
        instance = deserialized(name="John", age=30)
        self.assertEqual(instance.name, "John")
        self.assertEqual(instance.age, 30)

    def test_mixed_descriptions(self):
        """Test models with mixed field descriptions (some with, some without)."""
        original = MixedModel
        
        # Serialize and deserialize
        serialized = serialize_base_model(original)
        deserialized = deserialize_base_model(serialized)
        deserialized_schema = deserialized.model_json_schema()
        
        # Check mixed descriptions
        self.assertIsNone(deserialized_schema['properties']['name'].get('description'))
        self.assertIsNone(deserialized_schema['properties']['age'].get('description'))
        self.assertEqual(
            deserialized_schema['properties']['email'].get('description'),
            "User's email address"
        )
        
        # Test that instances can be created
        instance = deserialized(name="Jane", age=25, email="jane@example.com")
        self.assertEqual(instance.name, "Jane")
        self.assertEqual(instance.age, 25)
        self.assertEqual(instance.email, "jane@example.com")
