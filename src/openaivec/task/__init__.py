"""Task module for OpenAI Vector operations.

This module provides pre-configured tasks and task models for common operations
such as multilingual translation and other AI-powered tasks.

Classes:
    PreparedTask: A model representing a prepared task with configuration.

Constants:
    MULTILINGUAL_TRANSLATION_TASK: Pre-configured task for multilingual translation.
"""
from .tasks import MULTILINGUAL_TRANSLATION_TASK
from .model import PreparedTask

__all__ = [
    "MULTILINGUAL_TRANSLATION_TASK",
    "PreparedTask",
]