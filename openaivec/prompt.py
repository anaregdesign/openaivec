import difflib
from logging import Logger, getLogger
from typing import List
from xml.etree import ElementTree

from openai import OpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel

_logger: Logger = getLogger(__name__)

enhance_prompt: str = """
<SystemMessage>
    <Instructions>
        <!--
          This system message is designed to improve a JSON-based prompt.
          Follow the steps below in order.
        -->

        <!-- Step 1: Overall Quality Improvement -->
        <Instruction>
            Receive the prompt in JSON format (with fields "purpose", "cautions", "examples",
            and optionally "advices"). Improve its quality by:
            - Ensuring no logical contradictions or ambiguities exist in the entire input.
            - Always respond in the same linguistic language as the input.
            - Correcting or refining the sentences (while preserving the original intent).
            - In the "purpose" field: clearly describe the input semantics, output semantics, and the main goal.
            - In the "cautions" field: gather common points or edge cases found in "examples."
            - In the "examples" field: enhance the examples to cover a wide range of scenarios.
                - "source" and "result" must correspond one-to-one.
                - Provide at least 5 examples for better coverage.
            - If the "advices" field is present: use them to refine or adjust the "purpose", "examples", and "cautions" fields where necessary.
        </Instruction>

        <!-- Step 2: Improve the "purpose" field -->
        <Instruction>
            Make the "purpose" field more concise, clearer, and more informative.
            Specifically:
            - Clearly describe the input semantics (e.g., "Expects a product name or instruction sentence").
            - Clearly describe the output semantics (e.g., "Returns a category or expected issues").
            - Briefly explain the main goal or usage scenario of the prompt.
            - If the "advices" field is present, consider incorporating its insights into the "purpose" field.
        </Instruction>

        <!-- Step 3: Improve the "cautions" field -->
        <Instruction>
            Examine the examples. If there are common patterns or edge cases (exceptions),
            incorporate them as necessary into the "cautions" field to help the end user
            avoid pitfalls. If no changes are needed, keep them as is.
            Review the "examples" to ensure that no redundant cautions remain.
            If the "advices" field is present, reflect any relevant suggestions into the "cautions" field,
            ensuring consistency with the other fields and the overall intent.
        </Instruction>

        <!-- Step 4: Improve the "examples" field -->
        <Instruction>
            - Add or refine as many "examples" as needed based on the "purpose" and "cautions."
            - Remove any duplicate "source" values; each "source" must be unique across all examples.
            - If "advices" is present, use them to refine or adjust existing "examples" ensuring consistency throughout.
        </Instruction>

        <!-- Step 5: Resolve contradictions, ambiguities or redundancies -->
        <Instruction>
            Ensure that "purpose," "cautions," and "examples" remain consistent in the final output.
            If some contradictions, ambiguities or redundancies are found,
            add solutions that are as detailed and specific as possible to the "advices" field.
        </Instruction>
    </Instructions>

    <Examples>
        <!-- without any changes -->
        <Example>
            <Input>
                {
                    "purpose": "<some_purpose>",
                    "cautions": ["<caution1>", "<caution2>"],
                    "examples": [
                        {
                            "source": "<source1>",
                            "result": "<result1>"
                        },
                        {
                            "source": "<source2>",
                            "result": "<result2>"
                        },
                        {
                            "source": "<source3>",
                            "result": "<result3>"
                        },
                        {
                            "source": "<source4>",
                            "result": "<result4>"
                        },
                        {
                            "source": "<source5>",
                            "result": "<result5>"
                        }
                    ],
                    "advices": ["<advice1>", "<advice2>"]
                }
            </Input>
            <Output>
                {
                    "purpose": "<improved_purpose>",
                    "cautions": ["<caution1>", "<new_caution1>"],
                    "examples": [
                        {
                            "source": "<source1>",
                            "result": "<result1>"
                        },
                        {
                            "source": "<source2>",
                            "result": "<result2>"
                        },
                        {
                            "source": "<source3>",
                            "result": "<result3>"
                        },
                        {
                            "source": "<source4>",
                            "result": "<result4>"
                        },
                        {
                            "source": "<source5>",
                            "result": "<result5>"
                        },
                        {
                            "source": "<new_source1>",
                            "result": "<new_result1>"
                        },
                        {
                            "source": "<new_source2>",
                            "result": "<new_result2>"
                        }
                    ],
                    "advices": ["<new_advice1>", "<new_advice2>"]
                }
            </Output>
        </Example>
    </Examples>
</SystemMessage>
"""


class Example(BaseModel):
    source: str
    result: str


class FewShotPrompt(BaseModel):
    purpose: str
    cautions: List[str]
    examples: List[Example]
    advices: List[str]


class FewShotPromptBuilder:
    _prompt: FewShotPrompt

    def __init__(self):
        self._prompt = FewShotPrompt(purpose="", cautions=[], examples=[], advices=[])

    @classmethod
    def of(cls, prompt: FewShotPrompt) -> "FewShotPromptBuilder":
        builder = cls()
        builder._prompt = prompt
        return builder

    def purpose(self, purpose: str) -> "FewShotPromptBuilder":
        self._prompt.purpose = purpose
        return self

    def caution(self, caution: str) -> "FewShotPromptBuilder":
        if self._prompt.cautions is None:
            self._prompt.cautions = []
        self._prompt.cautions.append(caution)
        return self

    def example(self, source: str, result: str) -> "FewShotPromptBuilder":
        if self._prompt.examples is None:
            self._prompt.examples = []
        self._prompt.examples.append(Example(source=source, result=result))
        return self

    def enhance(
        self, client: OpenAI, model_name: str, temperature: float = 0, top_p: float = 1
    ) -> "FewShotPromptBuilder":

        # At least 5 examples are required to enhance the prompt.
        if len(self._prompt.examples) < 5:
            raise ValueError("At least 5 examples are required to enhance the prompt.")

        completion: ParsedChatCompletion[FewShotPrompt] = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": enhance_prompt},
                {
                    "role": "user",
                    "content": self._prompt.model_dump_json(),
                },
            ],
            temperature=temperature,
            top_p=top_p,
            response_format=FewShotPrompt,
        )
        self._prompt = completion.choices[0].message.parsed
        return self

    def improve(self, client: OpenAI, model_name: str, max_iter: int = 5) -> "FewShotPromptBuilder":
        for i in range(max_iter):
            _logger.info("Iteration %d", i + 1)
            original: str = self.build()
            self.enhance(client, model_name)
            self._warn_advices()
            improved: str = self.build()

            if original == improved:
                _logger.info("No further improvement.")
                break
            else:
                lines1 = original.splitlines()
                lines2 = improved.splitlines()
                diff = difflib.unified_diff(lines1, lines2, fromfile="before", tofile="after", lineterm="")
                for line in diff:
                    _logger.info(line)
        return self

    def _validate(self):
        # Validate that 'purpose' and 'examples' are not empty.
        if not self._prompt.purpose:
            raise ValueError("Purpose is required.")
        if not self._prompt.examples or len(self._prompt.examples) == 0:
            raise ValueError("At least one example is required.")

    def _warn_advices(self):
        if self._prompt.advices:
            for advice in self._prompt.advices:
                _logger.warning("Advice: %s", advice)

    def get_object(self) -> FewShotPrompt:
        self._validate()
        return self._prompt

    def build(self) -> str:
        self._validate()
        return self.build_xml()

    def build_json(self, **kwargs) -> str:
        self._validate()

        return self._prompt.model_dump_json(**kwargs)

    def build_xml(self) -> str:
        self._validate()

        prompt_dict = self._prompt.model_dump()
        root = ElementTree.Element("Prompt")

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

        ElementTree.indent(root, level=0)

        return ElementTree.tostring(root, encoding="unicode")
