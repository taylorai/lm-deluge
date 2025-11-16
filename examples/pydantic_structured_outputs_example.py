#!/usr/bin/env python3
"""Example demonstrating Pydantic model support for structured outputs."""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

from lm_deluge import LLMClient


# Define your Pydantic models
class Person(BaseModel):
    """A person with contact information."""

    name: str
    age: int = Field(ge=0, le=150, description="Age in years")
    email: str
    occupation: Optional[str] = None


class Address(BaseModel):
    """A physical address."""

    street: str
    city: str
    state: str
    zip_code: str


class Contact(BaseModel):
    """A contact with full details."""

    person: Person
    address: Address
    phone_numbers: List[str]


class TaskPriority(BaseModel):
    """A task with priority."""

    title: str
    description: str
    priority: Literal["low", "medium", "high", "urgent"]
    estimated_hours: float = Field(ge=0, description="Estimated hours to complete")


def example_simple_extraction():
    """Example: Extract person information using a Pydantic model."""
    client = LLMClient("gpt-4o-mini")

    text = """
    John Smith is a 35-year-old software engineer living in San Francisco.
    You can reach him at john.smith@example.com.
    """

    # Pass the Pydantic model directly to output_schema!
    responses = client.process_prompts_sync(
        prompts=[f"Extract the person information from this text:\n\n{text}"],
        output_schema=Person,  # ← Pydantic model, not a dict!
        return_completions_only=True,
    )

    print("Extracted person:")
    print(responses[0])


def example_nested_extraction():
    """Example: Extract nested contact information."""
    client = LLMClient("claude-4.5-sonnet")

    text = """
    Contact: Jane Doe, 28 years old, data scientist at TechCorp.
    Email: jane.doe@techcorp.com
    Address: 123 Main St, Boston, MA 02101
    Phone: 555-0123, 555-0124
    """

    responses = client.process_prompts_sync(
        prompts=[f"Extract the contact information:\n\n{text}"],
        output_schema=Contact,  # Nested Pydantic model
        return_completions_only=True,
    )

    print("Extracted contact:")
    print(responses[0])


def example_with_constraints():
    """Example: Use a model with field constraints."""
    client = LLMClient("gpt-4o-mini")

    # The constraints (ge, le, etc.) will be moved to field descriptions
    # so the model can still follow them, even though they're not enforced
    # by the JSON schema grammar
    responses = client.process_prompts_sync(
        prompts=["Create a high-priority task for code review"],
        output_schema=TaskPriority,
        return_completions_only=True,
    )

    print("Generated task:")
    print(responses[0])


def example_validation():
    """Example: Validate the response using Pydantic."""
    import json
    from pydantic import ValidationError

    client = LLMClient("gpt-4o-mini")

    responses = client.process_prompts_sync(
        prompts=["Generate a person: Alice, 30, alice@example.com"],
        output_schema=Person,
        return_completions_only=True,
    )

    # Parse and validate the response
    try:
        response_text = responses[0]
        if response_text and isinstance(response_text, str):
            person_data = json.loads(response_text)
            person = Person(**person_data)
            print(f"Valid person: {person.name}, {person.age} years old")
        else:
            print("No response received")
    except ValidationError as e:
        print(f"Validation error: {e}")


def example_traditional_dict_still_works():
    """Example: Traditional dict schemas still work."""
    client = LLMClient("gpt-4o-mini")

    # You can still use dict schemas if you prefer
    schema_dict = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "key_points"],
    }

    responses = client.process_prompts_sync(
        prompts=["Summarize the benefits of structured outputs"],
        output_schema=schema_dict,  # Dict schema
        return_completions_only=True,
    )

    print("Summary:")
    print(responses[0])


if __name__ == "__main__":
    print("=" * 60)
    print("Pydantic Structured Outputs Examples")
    print("=" * 60)
    print()

    print("1. Simple extraction with Pydantic model")
    print("-" * 60)
    example_simple_extraction()
    print()

    print("2. Nested extraction")
    print("-" * 60)
    example_nested_extraction()
    print()

    print("3. Model with field constraints")
    print("-" * 60)
    example_with_constraints()
    print()

    print("4. Response validation")
    print("-" * 60)
    example_validation()
    print()

    print("5. Traditional dict schemas still work")
    print("-" * 60)
    example_traditional_dict_still_works()
    print()

    print("=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
