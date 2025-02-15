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
            - Correcting or refining the language (but preserving the original language).
            - At least 5 examples are required to enhance the prompt.
        </Instruction>

        <!-- Step 2: Improve the "purpose" field -->
        <Instruction>
            Make the "purpose" field more concise, clearer, and more informative.
            Specifically:
            - Clearly describe the input semantics (e.g., "Expects a name of product or instruction sentence").
            - Clearly describe the output semantics (e.g., "Returns a category, or expected troubles").
            - Explain the main goal or usage scenario of the prompt in a concise manner.
        </Instruction>

        <!-- Step 3: Analyze the "examples" field -->
        <Instruction>
            Examine the examples. If there are common patterns or edge cases (exceptions),
            incorporate them as necessary into the "cautions" field to help the end user
            avoid pitfalls. If no changes are needed, keep them as is.
        </Instruction>

        <!-- Step 4: Handle the presence or absence of "advices" -->
        <Instruction>
            - If "advices" is not present in the input:
                1. Preserve the existing "cautions" and "examples" fields.
                2. Add as many similar "examples" as possible according the "purpose" and "cautions".
            - If "advices" is present in the input:
                1. Use the provided "advices" to refine or adjust "examples" and/or "cautions".
        </Instruction>

        <!-- Step 5: Address contradictions or ambiguities -->
        <Instruction>
            If you discover any contradictions or ambiguities within the "examples" or "cautions",
            add them as new entries in the "advices" field, along with proposed possible alternatives.
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
                    ]
                }
            </Input>
            <Output>
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
                    ]
                }
            </Output>
        </Example>
    
        <!-- with improved purpose -->
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
                    ]
                }
            </Input>
            <Output>
                {
                    "purpose": "<improved_purpose>",
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
                    ]
                }
            </Output>
        </Example>
    
        <!-- with additional cautions -->
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
                    ]
                }
            </Input>
            <Output>
                {
                    "purpose": "<improved_purpose>",
                    "cautions": [
                        "<caution1>",
                        "<caution2>",
                        "<additional_caution1>",
                        "<additional_caution2>"
                    ],
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
                    ]
                }
            </Output>
        </Example>
    
        <!-- with additional examples -->
        <Example>
            <Input>
                {
                    "purpose": "<some_purpose>",
                    "cautions": ["<caution1>"],
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
                    ]
                }
            </Input>
            <Output>
                {
                    "purpose": "<improved_purpose>",
                    "cautions": [
                        "<caution1>"
                    ],
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
                            "source": "<additional_source1>",
                            "result": "<additional_result1>"
                        },
                        {
                            "source": "<additional_source2>",
                            "result": "<additional_result2>"
                        }
                    ]
                }
            </Output>
        </Example>
    
        <!-- with additional examples and cautions -->
        <Example>
            <Input>
                {
                    "purpose": "<some_purpose>",
                    "cautions": [],
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
                    ]
                }
            </Input>
            <Output>
                {
                    "purpose": "<improved_purpose>",
                    "cautions": [
                        "<additional_caution1>",
                        "<additional_caution2>"
                    ],
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
                            "source": "<additional_source1>",
                            "result": "<additional_result1>"
                        },
                        {
                            "source": "<additional_source2>",
                            "result": "<additional_result2>"
                        }
                    ]
                }
            </Output>
        </Example>
    
        <!-- with advices -->
        <Example>
            <Input>
                {
                    "purpose": "<some_purpose>",
                    "cautions": ["<caution1>"],
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
                    ]
                }
            </Input>
            <Output>
                {
                    "purpose": "<improved_purpose>",
                    "cautions": [
                        "<caution1>"
                    ],
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
                    "advices": [
                        "<advice1>",
                        "<advice2>"
                    ]
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
    def __init__(self):
        self._prompt = FewShotPrompt(purpose="", cautions=[], examples=[], advices=[])

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
        self._warn_advices()
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
