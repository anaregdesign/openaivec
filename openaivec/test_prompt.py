import json
import logging
import unittest
from xml.etree import ElementTree

from openaivec.prompt import FewShotPromptBuilder

logging.basicConfig(level=logging.INFO, force=True)


class TestAtomicPromptBuilder(unittest.TestCase):
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
        builder = FewShotPromptBuilder()
        builder.purpose("Test Purpose")
        builder.example("source1", "result1")
        builder.caution("Check input")
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

        self.assertEqual(root.tag, "AtomicPrompt")

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



