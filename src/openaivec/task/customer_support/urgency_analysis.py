"""Urgency analysis task for customer support.

This module provides a predefined task for analyzing the urgency level of customer
inquiries to help prioritize support queue and response times.

Example:
    Basic usage with BatchResponses:
    
    ```python
    from openai import OpenAI
    from openaivec.responses import BatchResponses
    from openaivec.task import customer_support
    
    client = OpenAI()
    analyzer = BatchResponses.of_task(
        client=client,
        model_name="gpt-4o-mini",
        task=customer_support.URGENCY_ANALYSIS
    )
    
    inquiries = [
        "URGENT: My website is down and I'm losing customers!",
        "Can you help me understand how to use the new feature?",
        "I haven't received my order from last week"
    ]
    analyses = analyzer.parse(inquiries)
    
    for analysis in analyses:
        print(f"Urgency Level: {analysis.urgency_level}")
        print(f"Score: {analysis.urgency_score}")
        print(f"Response Time: {analysis.response_time}")
        print(f"Escalation: {analysis.escalation_required}")
    ```

    With pandas integration:
    
    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import customer_support
    
    df = pd.DataFrame({"inquiry": [
        "URGENT: My website is down and I'm losing customers!",
        "Can you help me understand how to use the new feature?",
        "I haven't received my order from last week"
    ]})
    df["urgency"] = df["inquiry"].ai.task(customer_support.URGENCY_ANALYSIS)
    
    # Extract urgency components
    extracted_df = df.ai.extract("urgency")
    print(extracted_df[["inquiry", "urgency_urgency_level", "urgency_urgency_score", "urgency_response_time"]])
    ```

Attributes:
    URGENCY_ANALYSIS (PreparedTask): A prepared task instance 
        configured for urgency analysis with temperature=0.0 and 
        top_p=1.0 for deterministic output.
"""

from typing import List
from pydantic import BaseModel, Field

from openaivec.task.model import PreparedTask

__all__ = ["URGENCY_ANALYSIS"]


class UrgencyAnalysis(BaseModel):
    urgency_level: str = Field(description="Urgency level: critical, high, medium, low")
    urgency_score: float = Field(description="Urgency score from 0.0 (not urgent) to 1.0 (extremely urgent)")
    response_time: str = Field(description="Recommended response time: immediate, within_1_hour, within_4_hours, within_24_hours, within_3_days")
    escalation_required: bool = Field(description="Whether this inquiry requires escalation to management")
    urgency_indicators: List[str] = Field(description="Specific words or phrases that indicate urgency")
    business_impact: str = Field(description="Potential business impact: none, low, medium, high, critical")
    customer_tier: str = Field(description="Inferred customer tier: enterprise, premium, standard, basic")
    reasoning: str = Field(description="Brief explanation of urgency assessment")


URGENCY_ANALYSIS = PreparedTask(
    instructions="""Analyze the urgency level of the customer inquiry based on language, content, and context.

Urgency Levels:
- critical: Service outages, security breaches, data loss, system failures affecting business operations
- high: Account locked, payment failures, urgent deadlines, angry customers, revenue-impacting issues
- medium: Feature not working, delivery delays, billing questions, moderate customer frustration
- low: General questions, feature requests, feedback, compliments, minor issues

Response Times:
- immediate: Critical issues requiring immediate attention (within 15 minutes)
- within_1_hour: High priority issues that need quick resolution
- within_4_hours: Medium priority issues during business hours
- within_24_hours: Standard response time for regular inquiries
- within_3_days: Low priority issues that can wait for next business cycle

Business Impact:
- critical: Major revenue loss, security risk, regulatory compliance issues
- high: Customer churn risk, operational disruption, reputation damage
- medium: Customer satisfaction impact, minor operational issues
- low: Informational requests, minor inconveniences
- none: Compliments, general feedback, non-urgent information

Customer Tier (inferred from language and context):
- enterprise: Large contracts, multiple users, business-critical usage
- premium: Paid plans, professional use, higher expectations
- standard: Regular paid users, normal expectations
- basic: Free users, casual usage, lower priority

Look for urgency indicators like:
- Explicit urgency words: "urgent", "emergency", "critical", "ASAP"
- Emotional language: "frustrated", "angry", "disappointed"
- Business impact: "losing money", "customers complaining", "deadline"
- Time pressure: "today", "immediately", "right now"
- Escalation language: "manager", "supervisor", "cancel subscription"

Provide detailed analysis with reasoning for urgency assessment.""",
    response_format=UrgencyAnalysis,
    temperature=0.0,
    top_p=1.0
)