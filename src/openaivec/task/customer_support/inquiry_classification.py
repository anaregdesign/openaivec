"""Inquiry classification task for customer support.

This module provides a predefined task for classifying customer inquiries into
different categories to help route them to the appropriate support team.

Example:
    Basic usage with BatchResponses:
    
    ```python
    from openai import OpenAI
    from openaivec.responses import BatchResponses
    from openaivec.task import customer_support
    
    client = OpenAI()
    classifier = BatchResponses.of_task(
        client=client,
        model_name="gpt-4o-mini",
        task=customer_support.INQUIRY_CLASSIFICATION
    )
    
    inquiries = [
        "I can't log into my account",
        "When will my order arrive?",
        "I want to cancel my subscription"
    ]
    classifications = classifier.parse(inquiries)
    
    for classification in classifications:
        print(f"Category: {classification.category}")
        print(f"Subcategory: {classification.subcategory}")
        print(f"Confidence: {classification.confidence}")
        print(f"Routing: {classification.routing}")
    ```

    With pandas integration:
    
    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import customer_support
    
    df = pd.DataFrame({"inquiry": [
        "I can't log into my account",
        "When will my order arrive?",
        "I want to cancel my subscription"
    ]})
    df["classification"] = df["inquiry"].ai.task(customer_support.INQUIRY_CLASSIFICATION)
    
    # Extract classification components
    extracted_df = df.ai.extract("classification")
    print(extracted_df[["inquiry", "classification_category", "classification_subcategory", "classification_confidence"]])
    ```

Attributes:
    INQUIRY_CLASSIFICATION (PreparedTask): A prepared task instance 
        configured for inquiry classification with temperature=0.0 and 
        top_p=1.0 for deterministic output.
"""

from typing import List
from pydantic import BaseModel, Field

from openaivec.task.model import PreparedTask

__all__ = ["INQUIRY_CLASSIFICATION"]


class InquiryClassification(BaseModel):
    category: str = Field(description="Primary category: technical, billing, product, shipping, account, general")
    subcategory: str = Field(description="Specific subcategory within the primary category")
    confidence: float = Field(description="Confidence score for classification (0.0-1.0)")
    routing: str = Field(description="Recommended routing: tech_support, billing_team, product_team, shipping_team, account_management, general_support")
    keywords: List[str] = Field(description="Key terms that influenced the classification")
    priority: str = Field(description="Suggested priority level: low, medium, high, urgent")


INQUIRY_CLASSIFICATION = PreparedTask(
    instructions="""Classify the customer inquiry into the appropriate category and subcategory. 

Categories and subcategories:
- technical: login_issues, password_reset, app_crashes, connectivity_problems, feature_not_working
- billing: payment_failed, invoice_questions, refund_request, pricing_inquiry, subscription_changes
- product: feature_request, product_information, compatibility_questions, how_to_use, bug_reports
- shipping: delivery_status, shipping_address, delivery_issues, tracking_number, expedited_shipping
- account: account_creation, profile_updates, account_deletion, data_export, privacy_settings
- general: compliments, complaints, feedback, partnership_inquiry, other

Routing options:
- tech_support: Technical issues requiring engineering expertise
- billing_team: Payment and subscription related issues
- product_team: Product questions and feature requests
- shipping_team: Delivery and logistics issues
- account_management: Account-related requests
- general_support: General inquiries and feedback

Priority levels:
- urgent: Account locked, payment failures, service outages
- high: Login issues, delivery problems, billing disputes
- medium: Feature requests, general questions, feedback
- low: Information requests, compliments, minor issues

Provide classification with confidence score, routing recommendation, relevant keywords, and priority level.""",
    response_format=InquiryClassification,
    temperature=0.0,
    top_p=1.0
)