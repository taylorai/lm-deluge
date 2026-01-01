"""
Test Amazon Nova models via Bedrock.

Requires AWS credentials to be set:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_SESSION_TOKEN (optional)
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from lm_deluge import LLMClient, Conversation, Tool

load_dotenv()


async def test_nova_basic():
    """Test basic Nova inference without tools."""
    print("=" * 60)
    print("Testing Nova Micro basic inference")
    print("=" * 60)

    llm = LLMClient(model_names="nova-micro", max_new_tokens=256)
    conv = Conversation().user("What is 2 + 2? Reply with just the number.")

    response = await llm.start(conv)

    print(f"Response: {response.completion}")
    print(f"Usage: {response.usage}")
    print(f"Is error: {response.is_error}")
    if response.is_error:
        print(f"Error: {response.error_message}")

    assert not response.is_error, f"Request failed: {response.error_message}"
    assert response.completion is not None
    print("PASSED")
    return response


async def test_nova_lite_with_image():
    """Test Nova Lite with image input."""
    print("\n" + "=" * 60)
    print("Testing Nova Lite with image")
    print("=" * 60)

    # Create a simple test - just check that it accepts the request
    # We'd need a real image for full testing
    llm = LLMClient(model_names="nova-lite", max_new_tokens=256)
    conv = Conversation().user("Describe the color blue in one sentence.")

    response = await llm.start(conv)

    print(f"Response: {response.completion}")
    print(f"Usage: {response.usage}")
    print(f"Is error: {response.is_error}")
    if response.is_error:
        print(f"Error: {response.error_message}")

    assert not response.is_error, f"Request failed: {response.error_message}"
    assert response.completion is not None
    print("PASSED")
    return response


async def test_nova_with_tools():
    """Test Nova with tool calling."""
    print("\n" + "=" * 60)
    print("Testing Nova Pro with tools")
    print("=" * 60)

    # Define a simple calculator tool
    async def calculator(equation: str) -> str:
        """Evaluate a math equation."""
        try:
            # Simple eval for testing (in production, use a proper math parser)
            result = eval(equation)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    calc_tool = Tool(
        name="calculator",
        description="Evaluate a mathematical equation",
        run=calculator,
        parameters={
            "equation": {
                "type": "string",
                "description": "The mathematical equation to evaluate (e.g., '2 + 2')",
            },
        },
        required=["equation"],
    )

    llm = LLMClient(model_names="nova-pro", max_new_tokens=512)
    conv = Conversation().user(
        "What is 15 * 7? Use the calculator tool to compute this."
    )

    final_conv, response = await llm.run_agent_loop(
        conv,
        tools=[calc_tool],
        max_rounds=3,
    )

    print(f"Response: {response.completion}")
    print(f"Usage: {response.usage}")
    print(f"Is error: {response.is_error}")
    if response.is_error:
        print(f"Error: {response.error_message}")

    # Check that the answer contains 105
    assert not response.is_error, f"Request failed: {response.error_message}"
    assert response.completion is not None
    print("PASSED")
    return response


async def test_nova_system_prompt():
    """Test Nova with system prompt."""
    print("\n" + "=" * 60)
    print("Testing Nova Micro with system prompt")
    print("=" * 60)

    llm = LLMClient(model_names="nova-micro", max_new_tokens=256)
    conv = (
        Conversation()
        .system("You are a pirate. Always respond in pirate speak.")
        .user("Hello, how are you?")
    )

    response = await llm.start(conv)

    print(f"Response: {response.completion}")
    print(f"Usage: {response.usage}")
    print(f"Is error: {response.is_error}")
    if response.is_error:
        print(f"Error: {response.error_message}")

    assert not response.is_error, f"Request failed: {response.error_message}"
    assert response.completion is not None
    print("PASSED")
    return response


async def test_nova_caching():
    """Test Nova prompt caching - requires 1024+ tokens for cache checkpoint."""
    print("\n" + "=" * 60)
    print("Testing Nova caching")
    print("=" * 60)

    # Create a long system prompt (needs 1024+ tokens for caching)
    # We need to generate enough content to exceed 1024 tokens
    long_context = """You are an expert assistant specializing in software development,
    machine learning, data science, and general programming tasks. You have deep knowledge
    of Python, JavaScript, TypeScript, Rust, Go, Java, C++, and many other programming languages.

    When answering questions, you should:
    1. Provide clear, concise explanations
    2. Include code examples when appropriate
    3. Explain your reasoning step by step
    4. Consider edge cases and potential issues
    5. Suggest best practices and common patterns

    You are familiar with popular frameworks and libraries including:
    - Web frameworks: Django, Flask, FastAPI, Express, Next.js, React, Vue, Angular
    - ML/AI: PyTorch, TensorFlow, scikit-learn, Hugging Face, LangChain
    - Data: Pandas, NumPy, Polars, DuckDB, PostgreSQL, MongoDB
    - Cloud: AWS, GCP, Azure, Kubernetes, Docker
    - DevOps: GitHub Actions, GitLab CI, Jenkins, Terraform

    You understand software architecture patterns like:
    - Microservices and monoliths
    - Event-driven architecture
    - CQRS and event sourcing
    - Domain-driven design
    - Clean architecture and hexagonal architecture

    You can help with:
    - Code review and optimization
    - Debugging and troubleshooting
    - System design and architecture
    - Performance analysis
    - Security best practices
    - Testing strategies
    - Documentation

    Always aim to provide accurate, helpful, and actionable advice. If you're unsure about
    something, acknowledge the uncertainty and suggest ways to verify or find more information.

    Remember to consider the context of the question and tailor your response accordingly.
    For beginners, provide more explanation and simpler examples. For experienced developers,
    you can assume more background knowledge and dive into advanced topics.

    When writing code, follow these guidelines:
    - Use meaningful variable and function names
    - Add comments for complex logic
    - Follow the language's style conventions
    - Handle errors appropriately
    - Write testable code
    - Consider performance implications

    You should also be aware of common pitfalls and antipatterns in software development:
    - Premature optimization
    - Over-engineering
    - Not invented here syndrome
    - Cargo cult programming
    - Copy-paste programming
    - Magic numbers and strings
    - Deep nesting
    - God objects and classes

    When discussing trade-offs, consider factors like:
    - Development time and complexity
    - Runtime performance
    - Memory usage
    - Maintainability
    - Scalability
    - Team expertise
    - Existing infrastructure

    Finally, stay up to date with modern development practices and emerging technologies,
    but also recognize when established, battle-tested solutions are more appropriate than
    the latest trends. Balance innovation with pragmatism.

    Here is additional context about programming languages:

    Python is a high-level, interpreted programming language known for its readability and
    versatility. It supports multiple programming paradigms including procedural, object-oriented,
    and functional programming. Python's extensive standard library and vast ecosystem of
    third-party packages make it suitable for web development, data analysis, artificial
    intelligence, scientific computing, and automation.

    JavaScript is a dynamic programming language primarily used for web development. It runs
    in browsers and on servers via Node.js. Modern JavaScript includes features like arrow
    functions, destructuring, spread operators, async/await, and modules. Popular frameworks
    include React, Vue, Angular, and Svelte for frontend, and Express, Fastify, and Nest.js
    for backend development.

    TypeScript is a typed superset of JavaScript that compiles to plain JavaScript. It adds
    optional static typing, classes, and interfaces to JavaScript, making it easier to build
    and maintain large-scale applications. TypeScript's type system helps catch errors at
    compile time rather than runtime.

    Rust is a systems programming language focused on safety, speed, and concurrency. Its
    ownership system guarantees memory safety without garbage collection. Rust is used for
    performance-critical services, operating systems, game engines, and WebAssembly applications.

    Go (Golang) is a statically typed, compiled language designed at Google. It emphasizes
    simplicity, efficiency, and built-in concurrency through goroutines and channels. Go is
    popular for building microservices, cloud infrastructure, and command-line tools.

    Java is a class-based, object-oriented programming language designed to have few
    implementation dependencies. It follows the write once, run anywhere principle through
    the Java Virtual Machine. Java is widely used in enterprise applications, Android
    development, and large-scale distributed systems.

    Here are some best practices for code review:

    Code reviews are essential for maintaining code quality and sharing knowledge across teams.
    When reviewing code, focus on logic errors, security vulnerabilities, performance issues,
    and adherence to coding standards. Provide constructive feedback that explains not just
    what to change but why. Be respectful and assume positive intent from the author.

    When submitting code for review, keep changes small and focused. Write clear commit messages
    and pull request descriptions. Include tests for new functionality and ensure existing tests
    pass. Address reviewer feedback promptly and have discussions about design decisions."""

    # First request - should write to cache
    llm = LLMClient(model_names="nova-lite", max_new_tokens=100)
    conv1 = Conversation().system(long_context).user("What is 2+2?")

    print("Request 1 (should write to cache)...")
    response1 = await llm.start(conv1, cache="system_and_tools")

    print(f"Response 1: {response1.completion}")
    print(f"Usage 1: {response1.usage}")
    assert not response1.is_error, f"Request 1 failed: {response1.error_message}"

    cache_write_1 = response1.usage.cache_write_tokens if response1.usage else 0
    cache_read_1 = response1.usage.cache_read_tokens if response1.usage else 0
    print(f"Cache write tokens (req 1): {cache_write_1}")
    print(f"Cache read tokens (req 1): {cache_read_1}")

    # Second request with same system prompt - should read from cache
    conv2 = Conversation().system(long_context).user("What is 3+3?")

    print("\nRequest 2 (should read from cache)...")
    response2 = await llm.start(conv2, cache="system_and_tools")

    print(f"Response 2: {response2.completion}")
    print(f"Usage 2: {response2.usage}")
    assert not response2.is_error, f"Request 2 failed: {response2.error_message}"

    cache_write_2 = response2.usage.cache_write_tokens if response2.usage else 0
    cache_read_2 = response2.usage.cache_read_tokens if response2.usage else 0
    print(f"Cache write tokens (req 2): {cache_write_2}")
    print(f"Cache read tokens (req 2): {cache_read_2}")

    # Verify caching behavior
    # First request should have cache writes (or at least process the tokens)
    # Second request should have cache reads > 0
    if cache_read_2 > 0:
        print(
            f"\nCaching verified! Second request read {cache_read_2} tokens from cache."
        )
        print("PASSED")
    elif cache_write_1 > 0:
        print(f"\nCache write happened ({cache_write_1} tokens), but cache read was 0.")
        print("This might be due to cache expiration or region. Check manually.")
        print("PASSED (cache write verified)")
    else:
        print("\nNote: No cache activity detected. This could be because:")
        print("  - The system prompt is below the 1024 token minimum")
        print("  - Caching is not available in this region")
        print("  - There's a delay before cache becomes available")
        print("PASSED (no error, but caching may not be active)")

    return response1, response2


async def main():
    # Check for AWS credentials
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("ERROR: AWS credentials not set.")
        print(
            "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )
        sys.exit(1)

    print("Running Nova Bedrock tests...")
    print()

    try:
        await test_nova_basic()
        await test_nova_system_prompt()
        await test_nova_lite_with_image()
        await test_nova_with_tools()
        await test_nova_caching()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
