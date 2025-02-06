from logging import basicConfig, Handler, StreamHandler
from types import SimpleNamespace
from typing import List
from unittest import TestCase
from unittest.mock import MagicMock

from openai.types.chat import ParsedChatCompletion

from openaivec import VectorizedOpenAI
from openaivec.vectorize import Message, Response

_h: Handler = StreamHandler()

basicConfig(handlers=[_h], level="DEBUG")


def create_dummy_parse(messages: List[Message]) -> ParsedChatCompletion[Response]:
    """Creates a dummy ParsedChatCompletion based on the provided messages.

    Args:
        messages (List[Message]): A list of Message instances.

    Returns:
        ParsedChatCompletion[Response]: A dummy parsed chat completion containing a response.
    """
    response = Response(
        assistant_messages=[Message(id=i, text=f"response_of_{m.text}") for i, m in enumerate(messages)]
    )
    dummy_message = SimpleNamespace(parsed=response)
    dummy_choice = SimpleNamespace(message=dummy_message)
    dummy_completion = SimpleNamespace(choices=[dummy_choice])

    return dummy_completion


class TestVectorizedOpenAI(TestCase):
    def setUp(self):
        """Set up test fixtures for VectorizedOpenAI.

        Initializes a mock OpenAI client, assigns test parameters,
        and creates an instance of VectorizedOpenAI.
        """
        self.mock_openai = MagicMock()
        self.mock_openai.beta.chat.completions.parse = MagicMock()
        self.system_message = "system_message"
        self.model_name = "model_name"

        self.client = VectorizedOpenAI(
            client=self.mock_openai,
            model_name=self.model_name,
            system_message=self.system_message,
        )

    def test_predict(self):
        """Test the predict method of VectorizedOpenAI.

        Verifies that the predict method returns the expected responses for given user messages.
        """
        user_messages = ["user_message_{i}" for i in range(7)]
        expected = [f"response_of_{m}" for m in user_messages]

        dummy_completion = create_dummy_parse([Message(id=i, text=message) for i, message in enumerate(user_messages)])
        self.mock_openai.beta.chat.completions.parse.return_value = dummy_completion

        actual = self.client.predict(user_messages)

        self.assertEqual(actual, expected)
