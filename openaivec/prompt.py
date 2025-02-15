from typing import List
from xml.etree import ElementTree

from pydantic import BaseModel


class FewShotExample(BaseModel):
    source: str
    result: str


class AtomicPrompt(BaseModel):
    purpose: str
    cautions: List[str]
    examples: List[FewShotExample]


class AtomicPromptBuilder:
    def __init__(self):
        self._prompt = AtomicPrompt(purpose="", cautions=[], examples=[])

    def purpose(self, purpose: str) -> "AtomicPromptBuilder":
        self._prompt.purpose = purpose
        return self

    def caution(self, caution: str) -> "AtomicPromptBuilder":
        if self._prompt.cautions is None:
            self._prompt.cautions = []
        self._prompt.cautions.append(caution)
        return self

    def example(self, source: str, result: str) -> "AtomicPromptBuilder":
        if self._prompt.examples is None:
            self._prompt.examples = []
        self._prompt.examples.append(FewShotExample(source=source, result=result))
        return self

    def build(self) -> AtomicPrompt:
        # Validate that 'purpose' and 'examples' are not empty.
        if not self._prompt.purpose:
            raise ValueError("Purpose is required.")
        if not self._prompt.examples or len(self._prompt.examples) == 0:
            raise ValueError("At least one example is required.")
        return self._prompt

    def build_json(self, **kwargs) -> str:
        prompt = self.build()
        return prompt.model_dump_json(**kwargs)

    def build_xml(self) -> str:
        prompt = self.build()
        prompt_dict = prompt.model_dump()
        root = ElementTree.Element("AtomicPrompt")

        # Purpose (always output)
        purpose_elem = ElementTree.SubElement(root, "Purpose")
        purpose_elem.text = prompt_dict["purpose"]

        # Cautions (always output, even if empty)
        cautions_elem = ElementTree.SubElement(root, "Cautions")
        if prompt_dict.get("cautions"):
            for caution in prompt_dict["cautions"]:
                caution_elem = ElementTree.SubElement(cautions_elem, "Caution")
                caution_elem.text = caution

        # Examples (always output)
        examples_elem = ElementTree.SubElement(root, "Examples")
        for example in prompt_dict["examples"]:
            example_elem = ElementTree.SubElement(examples_elem, "Example")
            source_elem = ElementTree.SubElement(example_elem, "Source")
            source_elem.text = example.get("source")
            result_elem = ElementTree.SubElement(example_elem, "Result")
            result_elem.text = example.get("result")

        return ElementTree.tostring(root, encoding="utf-8")
