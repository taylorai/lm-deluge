"""Validate that Pydantic documentation examples are syntactically correct."""

import ast
import sys


def validate_code_block(code: str, description: str) -> bool:
    """Validate that a code block is syntactically correct Python."""
    try:
        ast.parse(code)
        print(f"✓ {description}: Syntax valid")
        return True
    except SyntaxError as e:
        print(f"✗ {description}: Syntax error - {e}")
        return False


# Example 1: Basic Pydantic structured output
example1 = """
from pydantic import BaseModel
from lm_deluge import LLMClient

class BugReport(BaseModel):
    summary: str
    priority: str  # "low", "medium", or "high"
    action_items: list[str]

client = LLMClient("gpt-4o-mini")

response = client.process_prompts_sync(
    ["Summarize this bug report and flag the priority."],
    output_schema=BugReport,
    show_progress=False,
)[0]

# Parse the response into your Pydantic model
data = BugReport.model_validate_json(response.completion)
print(data.summary, data.priority)
"""

# Example 2: Extract with Pydantic
example2 = """
from pydantic import BaseModel
from lm_deluge import LLMClient
from lm_deluge.llm_tools.extract import extract

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float
    vendor_name: str
    line_items: list[str]

client = LLMClient("gpt-4o-mini")

# Extract from multiple documents at once
documents = [
    "Invoice #12345\\nVendor: Acme Corp\\nTotal: $1,234.56\\n...",
    "Invoice #67890\\nVendor: TechCo\\nTotal: $987.65\\n...",
]

results = extract(
    inputs=documents,
    schema=Invoice,  # Pass your Pydantic model directly
    client=client,
    document_name="invoice",  # Used in the extraction prompt
    object_name="invoice data",  # Used in the extraction prompt
)

# Results are already parsed as dicts
for result in results:
    if result and "error" not in result:
        invoice = Invoice(**result)
        print(f"Invoice {invoice.invoice_number}: ${invoice.total_amount}")
"""

# Example 3: Advanced Pydantic features
example3 = """
from typing import Literal
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    '''Customer review of a product'''

    rating: int = Field(ge=1, le=5, description="Star rating from 1 to 5")
    sentiment: Literal["positive", "negative", "neutral"]
    product_name: str
    review_text: str
    would_recommend: bool
    tags: list[str] = Field(default_factory=list, description="Keywords describing the review")

    class Config:
        # Prevent the model from adding extra fields
        extra = "forbid"

client = LLMClient("claude-sonnet-4")

response = client.process_prompts_sync(
    ["Analyze this customer review: 'Great product! Works as advertised. 5 stars!'"],
    output_schema=ProductReview,
    show_progress=False,
)[0]

review = ProductReview.model_validate_json(response.completion)
print(f"Rating: {review.rating}/5, Sentiment: {review.sentiment}")
"""

# Example 4: Tools with Pydantic (from docs)
example4 = """
from lm_deluge.config import SamplingParams
from lm_deluge.tool import Tool

weather = Tool.from_function(get_weather)

responses = client.process_prompts_sync(
    ["Plan a weekend trip with weather calls and produce structured JSON."],
    tools=[weather],
    output_schema=schema,
    show_progress=False,
    sampling_params=[SamplingParams(strict_tools=True)],
)
"""


def main():
    print("Validating Pydantic documentation examples...\n")

    all_valid = True
    all_valid &= validate_code_block(example1, "Basic Pydantic structured output")
    all_valid &= validate_code_block(example2, "Extract with Pydantic")
    all_valid &= validate_code_block(example3, "Advanced Pydantic features")
    all_valid &= validate_code_block(example4, "Tools with structured outputs")

    print()
    if all_valid:
        print("✓ All documentation examples are syntactically valid!")
        return 0
    else:
        print("✗ Some documentation examples have syntax errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
