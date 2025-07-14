"""Task module for OpenAI Vector operations.

This module provides pre-configured tasks and task models for common operations
such as multilingual translation and other AI-powered NLP tasks.

Classes:
    PreparedTask: A model representing a prepared task with configuration.

Attributes:
    MULTILINGUAL_TRANSLATION: Pre-configured task for multilingual translation.
    MORPHOLOGICAL_ANALYSIS: Pre-configured task for morphological analysis including
        tokenization, part-of-speech tagging, and lemmatization.
    NAMED_ENTITY_RECOGNITION: Pre-configured task for named entity recognition that
        identifies and classifies named entities in text.
    SENTIMENT_ANALYSIS: Pre-configured task for sentiment analysis that analyzes
        sentiment and emotions in text.
    DEPENDENCY_PARSING: Pre-configured task for dependency parsing that analyzes
        syntactic dependencies between words in sentences.
    KEYWORD_EXTRACTION: Pre-configured task for keyword extraction that identifies
        important keywords and phrases from text.
"""
from .nlp import (
    MULTILINGUAL_TRANSLATION,
    MORPHOLOGICAL_ANALYSIS,
    NAMED_ENTITY_RECOGNITION,
    SENTIMENT_ANALYSIS,
    DEPENDENCY_PARSING,
    KEYWORD_EXTRACTION,
)
from .model import PreparedTask

__all__ = [
    "MULTILINGUAL_TRANSLATION",
    "MORPHOLOGICAL_ANALYSIS",
    "NAMED_ENTITY_RECOGNITION",
    "SENTIMENT_ANALYSIS",
    "DEPENDENCY_PARSING",
    "KEYWORD_EXTRACTION",
    "PreparedTask",
]