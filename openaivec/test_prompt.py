import json
import logging
import os
import unittest
from xml.etree import ElementTree

import langdetect
from openai import OpenAI, AzureOpenAI

from openaivec.prompt import FewShotPromptBuilder, FewShotPrompt

logging.basicConfig(level=logging.INFO, force=True)


class TestAtomicPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.client: OpenAI = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        )
        self.model_name: str = os.environ.get("AZURE_OPENAI_MODEL_NAME")

    def test_missing_purpose_raises_error(self):
        """Test that a ValueError is raised if 'purpose' is missing."""
        builder = FewShotPromptBuilder()
        builder.example("source1", "result1")
        with self.assertRaises(ValueError) as context:
            builder.build()
        self.assertEqual(str(context.exception), "Purpose is required.")

    def test_missing_examples_raises_error(self):
        """Test that a ValueError is raised if 'examples' is missing."""
        builder = FewShotPromptBuilder()
        builder.purpose("Test Purpose")
        with self.assertRaises(ValueError) as context:
            builder.build()
        self.assertEqual(str(context.exception), "At least one example is required.")

    def test_build_json_success(self):
        """Test successful JSON serialization when all required fields are set."""
        builder = FewShotPromptBuilder().purpose("Test Purpose").example("source1", "result1").caution("Check input")
        json_str = builder.build_json()

        # Parse the JSON string to verify its content
        data = json.loads(json_str)

        # Log the parsed JSON result
        reformatted: str = json.dumps(data, indent=2)
        logging.info("Parsed JSON: %s", reformatted)

        self.assertEqual(data["purpose"], "Test Purpose")
        self.assertEqual(data["cautions"], ["Check input"])
        self.assertIn("examples", data)
        self.assertEqual(len(data["examples"]), 1)
        self.assertEqual(data["examples"][0]["source"], "source1")
        self.assertEqual(data["examples"][0]["result"], "result1")

    def test_build_xml_success(self):
        """Test successful XML serialization when all required fields are set."""
        builder = FewShotPromptBuilder()
        builder.purpose("Test Purpose")
        builder.example("source1", "result1")
        builder.caution("Check input")
        xml_bytes = builder.build_xml()

        # Parse the XML to verify its structure and content
        root = ElementTree.fromstring(xml_bytes)
        ElementTree.indent(root, level=0)

        # Log the parsed XML result
        parsed_xml = ElementTree.tostring(root, encoding="unicode")
        logging.info("Parsed XML: %s", parsed_xml)

        self.assertEqual(root.tag, "Prompt")

        # Check the Purpose tag
        purpose_elem = root.find("Purpose")
        self.assertIsNotNone(purpose_elem)
        self.assertEqual(purpose_elem.text, "Test Purpose")

        # Check the Cautions tag
        cautions_elem = root.find("Cautions")
        self.assertIsNotNone(cautions_elem)
        caution_elem = cautions_elem.find("Caution")
        self.assertIsNotNone(caution_elem)
        self.assertEqual(caution_elem.text, "Check input")

        # Check the Examples tag
        examples_elem = root.find("Examples")
        self.assertIsNotNone(examples_elem)
        example_elem = examples_elem.find("Example")
        self.assertIsNotNone(example_elem)
        source_elem = example_elem.find("Source")
        self.assertIsNotNone(source_elem)
        self.assertEqual(source_elem.text, "source1")
        result_elem = example_elem.find("Result")
        self.assertIsNotNone(result_elem)
        self.assertEqual(result_elem.text, "result1")

    def test_improve(self):
        prompt: str = (
            FewShotPromptBuilder()
            .purpose("Return the smallest category that includes the given word")
            .caution("Never use proper nouns as categories")
            .example("Apple", "Fruit")
            .example("Car", "Vehicle")
            .example("Tokyo", "City")
            .example("Keiichi Sogabe", "Musician")
            .example("America", "Country")
            .example("United Kingdom", "Country")
            # Examples of countries
            .example("France", "Country")
            .example("Germany", "Country")
            .example("Brazil", "Country")
            # Examples of famous Americans
            .example("Elvis Presley", "Musician")
            .example("Marilyn Monroe", "Actor")
            .example("Michael Jordan", "Athlete")
            # Examples of American place names
            .example("New York", "City")
            .example("Los Angeles", "City")
            .example("Grand Canyon", "Natural Landmark")
            # Examples of everyday items
            .example("Toothbrush", "Hygiene Product")
            .example("Notebook", "Stationery")
            .example("Spoon", "Kitchenware")
            # Examples of company names
            .example("Google", "Company in USA")
            .example("Toyota", "Company in Japan")
            .example("Amazon", "Company in USA")
            # Examples of abstract concepts
            .example("Freedom", "Abstract Idea")
            .example("Happiness", "Emotion")
            .example("Justice", "Ethical Principle")
            # Steve Wozniak is not boring
            .example("Steve Wozniak", "is not boring")
            .improve(self.client, self.model_name, max_iter=5)
            .build()
        )

        # Log the parsed XML result
        logging.info("Parsed XML: %s", prompt)

    def test_few_examples_raise_error(self):
        """Test that a ValueError is raised if less than 5 examples are provided."""
        builder = (
            FewShotPromptBuilder().purpose("Test Purpose").example("source1", "result1").example("source2", "result2")
        )
        with self.assertRaises(ValueError) as context:
            builder.enhance(self.client, self.model_name)
        self.assertEqual(str(context.exception), "At least 5 examples are required to enhance the prompt.")

    def test_advice(self):
        """Test that a warning is raised if a contradiction is detected in the examples."""
        prompt: str = (
            FewShotPromptBuilder()
            .purpose("Return the smallest category that includes the given word")
            .example("Apple", "Fruit")
            .example("Apple", "Technology")
            .example("Apple", "Company")
            .example("Apple", "Color")
            .example("Apple", "Animal")
            .improve(self.client, self.model_name, max_iter=5)
            .build()
        )
        logging.info("Prompt: %s", prompt)

    def test_language_consistency_ja(self):
        """Test that linguistic language is preserved between input and output."""

        prompt_ja: FewShotPrompt = (
            FewShotPromptBuilder()
            .purpose("受け取った単語を含む最小のカテゴリ名を返してください")
            .caution("カテゴリ名に固有名詞を使用しないでください")
            .example("りんご", "果物")
            .example("パンダ", "クマ科")
            .example("東京", "都市")
            .example("ネコ", "ネコ科")
            .example("アメリカ", "国")
            .get_object()
        )

        improved_prompt_ja: FewShotPrompt = (
            FewShotPromptBuilder.of(prompt_ja).improve(self.client, self.model_name, max_iter=5).get_object()
        )

        # Assert that the linguistic language is preserved between input and output with langdetect
        self.assertEqual(langdetect.detect(prompt_ja.purpose), langdetect.detect(improved_prompt_ja.purpose))
