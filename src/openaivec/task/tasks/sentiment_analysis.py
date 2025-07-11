"""Sentiment analysis task for OpenAI API.

This module provides a predefined task for sentiment analysis that analyzes
sentiment and emotions in text using OpenAI's language models.

Example:
    Basic usage with BatchResponses:
    
    ```python
    from openai import OpenAI
    from openaivec.responses import BatchResponses
    from openaivec import task
    
    client = OpenAI()
    analyzer = BatchResponses.of_task(
        client=client,
        model_name="gpt-4o-mini",
        task=task.SENTIMENT_ANALYSIS_TASK
    )
    
    texts = ["I love this product!", "This is terrible and disappointing."]
    analyses = analyzer.parse(texts)
    
    for analysis in analyses:
        print(f"Sentiment: {analysis.sentiment}")
        print(f"Confidence: {analysis.confidence}")
        print(f"Emotions: {analysis.emotions}")
    ```

Attributes:
    SENTIMENT_ANALYSIS_TASK (PreparedTask): A prepared task instance 
        configured for sentiment analysis with temperature=0.0 and 
        top_p=1.0 for deterministic output.
"""

from typing import List
from openai import BaseModel
from pydantic import Field

from openaivec.task.model import PreparedTask

__all__ = ["SENTIMENT_ANALYSIS_TASK"]


class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score for sentiment (0.0-1.0)")
    emotions: List[str] = Field(description="Detected emotions (joy, sadness, anger, fear, surprise, disgust)")
    emotion_scores: List[float] = Field(description="Confidence scores for each emotion (0.0-1.0)")
    polarity: float = Field(description="Polarity score from -1.0 (negative) to 1.0 (positive)")
    subjectivity: float = Field(description="Subjectivity score from 0.0 (objective) to 1.0 (subjective)")


SENTIMENT_ANALYSIS_TASK = PreparedTask(
    instructions="Analyze the sentiment and emotions in the following text. Provide overall sentiment classification, confidence scores, detected emotions, polarity, and subjectivity measures.",
    response_format=SentimentAnalysis,
    temperature=0.0,
    top_p=1.0
)