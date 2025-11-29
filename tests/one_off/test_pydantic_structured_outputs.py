"""Test Pydantic structured outputs examples from documentation."""

from typing import Literal

from pydantic import BaseModel, Field

from lm_deluge import LLMClient
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.pipelines.extract import extract


def test_basic_pydantic_structured_output():
    """Test basic Pydantic model with structured outputs"""

    class BugReport(BaseModel):
        summary: str
        priority: str  # "low", "medium", or "high"
        action_items: list[str]

    client = LLMClient("gpt-4o-mini")

    response = client.process_prompts_sync(
        [
            "Bug: Login button doesn't work on mobile. Priority: high. Action items: Fix CSS, Test on iOS, Test on Android."
        ],
        output_schema=BugReport,
        show_progress=False,
    )[0]

    # Verify we got a response
    assert response is not None
    assert isinstance(response, APIResponse)
    assert response.completion is not None

    # Parse the response into Pydantic model
    data = BugReport.model_validate_json(response.completion)

    # Verify structure
    assert isinstance(data.summary, str)
    assert isinstance(data.priority, str)
    assert isinstance(data.action_items, list)
    assert len(data.action_items) > 0

    print("✓ Basic Pydantic structured output test passed")
    print(f"  Summary: {data.summary}")
    print(f"  Priority: {data.priority}")
    print(f"  Action items: {len(data.action_items)}")


def test_advanced_pydantic_features():
    """Test advanced Pydantic features with structured outputs"""

    class ProductReview(BaseModel):
        """Customer review of a product"""

        rating: int = Field(ge=1, le=5, description="Star rating from 1 to 5")
        sentiment: Literal["positive", "negative", "neutral"]
        product_name: str
        review_text: str
        would_recommend: bool
        tags: list[str] = Field(
            default_factory=list, description="Keywords describing the review"
        )

        class Config:
            extra = "forbid"

    client = LLMClient("gpt-4o-mini")

    response = client.process_prompts_sync(
        [
            "Review: 'Great wireless headphones! Sound quality is amazing and battery lasts all day. Totally worth the price. 5 stars!'"
        ],
        output_schema=ProductReview,
        show_progress=False,
    )[0]

    assert response is not None
    assert isinstance(response, APIResponse)
    assert response.completion is not None

    # Parse and validate
    review = ProductReview.model_validate_json(response.completion)

    # Verify constraints
    assert 1 <= review.rating <= 5
    assert review.sentiment in ["positive", "negative", "neutral"]
    assert isinstance(review.would_recommend, bool)
    assert isinstance(review.tags, list)

    print("✓ Advanced Pydantic features test passed")
    print(f"  Rating: {review.rating}/5")
    print(f"  Sentiment: {review.sentiment}")
    print(f"  Product: {review.product_name}")
    print(f"  Recommend: {review.would_recommend}")


def test_extract_with_pydantic():
    """Test extract() function with Pydantic models"""

    class Invoice(BaseModel):
        invoice_number: str
        total_amount: float
        vendor_name: str
        line_items: list[str]

    client = LLMClient("gpt-4o-mini")

    # Test with multiple documents
    documents = [
        "Invoice #INV-12345\nVendor: Acme Corporation\nTotal Amount: $1,234.56\nLine items:\n- Widget A: $500\n- Widget B: $734.56",
        "Invoice Number: INV-67890\nFrom: TechCo Industries\nTotal: $987.65\nItems:\n- Service Plan\n- Hardware",
    ]

    results = extract(
        inputs=documents,
        schema=Invoice,
        client=client,
        document_name="invoice",
        object_name="invoice data",
        show_progress=False,
    )

    # Verify we got results
    assert len(results) == 2

    # Verify each result
    for i, result in enumerate(results):
        assert result is not None and isinstance(result, dict)
        if "error" not in result:
            invoice = Invoice(**result)
            assert isinstance(invoice.invoice_number, str)
            assert isinstance(invoice.total_amount, float)
            assert isinstance(invoice.vendor_name, str)
            assert isinstance(invoice.line_items, list)
            assert len(invoice.line_items) > 0

            print(f"✓ Extract test {i + 1} passed")
            print(f"  Invoice: {invoice.invoice_number}")
            print(f"  Vendor: {invoice.vendor_name}")
            print(f"  Total: ${invoice.total_amount}")


if __name__ == "__main__":
    print("Testing Pydantic structured outputs...\n")

    test_basic_pydantic_structured_output()
    print()

    test_advanced_pydantic_features()
    print()

    test_extract_with_pydantic()
    print()

    print("All Pydantic structured output tests passed!")
