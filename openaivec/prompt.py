from typing import List
from xml.etree import ElementTree

from openai import OpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel

enhance_prompt: str = """
<SystemMessage>
    <Instructions>
        <Instruction>
            Receive the prompt and improve its quality.
        </Instruction>
        <Instruction>
            Improve the prompt sentence by making it more clear, concise, and informative.
        </Instruction>
        <Instruction>
            Identify common characteristics or exceptions in the examples and incorporate them 
            into the "cautions" field as needed.
        </Instruction>
        <Instruction>
            Keep the original elements of "cautions" and "examples" intact.
        </Instruction>
        <Instruction>
            Always preserve the original input language in your output.
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


class FewShotPromptBuilder:
    def __init__(self):
        self._prompt = FewShotPrompt(purpose="", cautions=[], examples=[])

    def purpose(self, purpose: str) -> "Few":
        self._prompt.purpose = purpose
        return self

    def caution(self, caution: str) -> "Few":
        if self._prompt.cautions is None:
            self._prompt.cautions = []
        self._prompt.cautions.append(caution)
        return self

    def example(self, source: str, result: str) -> "Few":
        if self._prompt.examples is None:
            self._prompt.examples = []
        self._prompt.examples.append(Example(source=source, result=result))
        return self

    def enhance(self, client: OpenAI, model_name: str, temperature: float = 0,
                top_p: float = 1) -> "Few":

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

    def build(self) -> FewShotPrompt:
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
