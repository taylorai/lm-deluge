"""
Real caching integration test that hits the Anthropic API.

This test requires:
1. ANTHROPIC_API_KEY environment variable to be set
2. Actual API credits (will use real tokens)
3. Minimum 1024 tokens to be cached (per Anthropic's requirements)

Run with: python tests/test_real_caching.py
"""

import os
import asyncio
from lm_deluge import LLMClient
from lm_deluge.prompt import Conversation, Message
from lm_deluge.tool import Tool


def create_long_system_message() -> str:
    """Create a system message that's definitely over 1024 tokens."""
    return """You are an expert software engineer, technical architect, and systems designer with over 15 years of experience building large-scale distributed systems. Your expertise spans the entire technology stack and software development lifecycle.

TECHNICAL EXPERTISE:

Programming Languages & Frameworks:
- Backend: Python (Django, Flask, FastAPI, Celery), Java (Spring Boot, Spring Cloud, JPA/Hibernate), Node.js (Express, NestJS, Fastify), Go (Gin, Echo, gRPC), Rust (Actix, Warp, Tokio), C# (.NET Core, ASP.NET, Entity Framework), Ruby (Rails, Sinatra), PHP (Laravel, Symfony), Scala (Play, Akka)
- Frontend: React (Redux, Context API, Next.js), Vue.js (Vuex, Nuxt.js), Angular (RxJS, NgRx), TypeScript, JavaScript (ES6+), HTML5, CSS3, SASS/LESS, Webpack, Vite
- Mobile: Swift/SwiftUI (iOS), Kotlin/Java (Android), React Native, Flutter, Xamarin, Ionic
- Data Science: Python (Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch), R, Julia, Spark, MLflow

Cloud Platforms & Services:
- AWS: EC2, ECS/EKS, Lambda, API Gateway, RDS, DynamoDB, S3, CloudFront, Route 53, VPC, IAM, CloudFormation, CDK
- Google Cloud: Compute Engine, GKE, Cloud Functions, Cloud Run, BigQuery, Firestore, Cloud Storage, Cloud CDN
- Azure: Virtual Machines, AKS, Functions, App Service, Cosmos DB, Blob Storage, Active Directory, ARM Templates
- Multi-cloud strategies, hybrid cloud architectures, cloud cost optimization

DevOps & Infrastructure:
- Containerization: Docker, Docker Compose, Kubernetes, Helm, OpenShift, containerd
- Infrastructure as Code: Terraform, Pulumi, AWS CDK, Azure ARM, Google Cloud Deployment Manager
- CI/CD: Jenkins, GitLab CI/CD, GitHub Actions, Azure DevOps, CircleCI, Travis CI, TeamCity
- Monitoring: Prometheus, Grafana, ELK Stack, Splunk, New Relic, DataDog, CloudWatch, Application Insights
- Configuration Management: Ansible, Chef, Puppet, SaltStack

Databases & Data Systems:
- Relational: PostgreSQL, MySQL, SQL Server, Oracle, advanced query optimization, indexing strategies
- NoSQL: MongoDB, Cassandra, Redis, DynamoDB, Elasticsearch, CouchDB, Neo4j
- Data Warehousing: Snowflake, BigQuery, Redshift, Databricks, Apache Spark, Hadoop ecosystem
- Real-time: Apache Kafka, RabbitMQ, Apache Pulsar, Redis Streams, AWS Kinesis

Software Architecture Patterns:
- Microservices architecture, service mesh (Istio, Linkerd), API gateway patterns
- Event-driven architecture, CQRS, Event Sourcing, Saga patterns
- Domain-driven design, hexagonal architecture, clean architecture
- Distributed systems patterns: Circuit breaker, bulkhead, timeout, retry with backoff
- Caching strategies: Redis, Memcached, CDN, application-level caching

Security & Compliance:
- Authentication & Authorization: OAuth 2.0, JWT, SAML, LDAP, Multi-factor authentication
- Security scanning, vulnerability assessment, penetration testing
- Compliance frameworks: GDPR, HIPAA, SOC 2, PCI DSS, ISO 27001
- Encryption at rest and in transit, key management, secrets management

METHODOLOGY & PRACTICES:

Development Practices:
- Test-driven development, behavior-driven development, unit testing, integration testing
- Code review processes, pair programming, mob programming
- Clean code principles, SOLID principles, design patterns
- Continuous refactoring, technical debt management
- Documentation strategies, API documentation, architectural decision records

Project Management & Agile:
- Scrum, Kanban, SAFe, Lean software development
- Requirements gathering, user story writing, acceptance criteria
- Sprint planning, retrospectives, stakeholder management
- Risk assessment, technical planning, capacity planning

Performance & Scalability:
- Load testing, stress testing, performance profiling
- Horizontal and vertical scaling strategies
- Database optimization, query performance tuning
- Caching strategies, CDN optimization
- Async programming, reactive programming patterns

COMMUNICATION GUIDELINES:

When providing technical solutions:
1. Always consider the business context and constraints
2. Provide multiple approaches with trade-offs analysis
3. Include specific implementation details and code examples
4. Consider scalability, maintainability, and security implications
5. Suggest appropriate testing strategies
6. Include monitoring and observability recommendations
7. Consider team skills and organizational readiness
8. Provide migration strategies for existing systems
9. Include cost considerations and resource requirements
10. Suggest incremental implementation approaches

Response Format:
- Start with a brief executive summary
- Provide detailed technical analysis
- Include practical implementation steps
- Suggest next steps and follow-up considerations
- Use clear headings and bullet points for readability
- Include relevant code snippets with explanations
- Provide links to additional resources when helpful"""


def create_comprehensive_tools() -> list[Tool]:
    """Create a comprehensive set of tools to ensure we hit the 1024 token minimum."""

    def search_code_repository(
        query: str, file_types: list[str] | None = None, max_results: int = 10
    ) -> str:
        """
        Search through a code repository for specific patterns, functions, or code snippets.

        Args:
            query: The search query (can be regex, function names, or plain text)
            file_types: List of file extensions to search (e.g., ['.py', '.js', '.java'])
            max_results: Maximum number of results to return

        Returns:
            JSON string containing search results with file paths, line numbers, and matching code
        """
        return f"Searched for '{query}' in repository"

    def analyze_code_quality(
        file_path: str, check_types: list[str] | None = None
    ) -> str:
        """
        Analyze code quality metrics for a specific file or directory.

        Args:
            file_path: Path to the file or directory to analyze
            check_types: Types of checks to perform (e.g., ['complexity', 'duplicates', 'security', 'performance'])

        Returns:
            JSON report containing quality metrics, issues found, and recommendations
        """
        return f"Analyzed code quality for {file_path}"

    def generate_documentation(
        file_path: str, doc_type: str, include_examples: bool = True
    ) -> str:
        """
        Generate comprehensive documentation for code files or modules.

        Args:
            file_path: Path to the file or module to document
            doc_type: Type of documentation ('api', 'readme', 'tutorial', 'reference')
            include_examples: Whether to include code examples in the documentation

        Returns:
            Generated documentation in markdown format
        """
        return f"Generated {doc_type} documentation for {file_path}"

    def run_tests(
        test_path: str, test_type: str = "unit", coverage: bool = True
    ) -> str:
        """
        Execute tests and return detailed results including coverage information.

        Args:
            test_path: Path to test files or directory
            test_type: Type of tests to run ('unit', 'integration', 'e2e', 'performance')
            coverage: Whether to include code coverage analysis

        Returns:
            JSON report with test results, coverage data, and performance metrics
        """
        return f"Executed {test_type} tests at {test_path}"

    def deploy_application(
        environment: str,
        config_overrides: dict | None = None,
        rollback_on_failure: bool = True,
    ) -> str:
        """
        Deploy application to specified environment with configuration management.

        Args:
            environment: Target environment ('development', 'staging', 'production')
            config_overrides: Dictionary of configuration values to override
            rollback_on_failure: Whether to automatically rollback on deployment failure

        Returns:
            Deployment status and details including URLs, health checks, and rollback info
        """
        return f"Deployed to {environment} environment"

    return [
        Tool.from_function(search_code_repository),
        Tool.from_function(analyze_code_quality),
        Tool.from_function(generate_documentation),
        Tool.from_function(run_tests),
        Tool.from_function(deploy_application),
    ]


async def test_real_caching_integration():
    """Test real caching with Anthropic API."""
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Skipping real caching test - ANTHROPIC_API_KEY not set")
        return

    print("🧪 Running real Anthropic caching integration test...")

    # Create client
    client = LLMClient.basic("claude-3.5-sonnet")

    # Create long system message and tools to ensure we hit 1024+ tokens
    system_msg = create_long_system_message()
    tools = create_comprehensive_tools()

    # Create initial conversation with system message
    conv1 = Conversation.system(system_msg).add(
        Message.user("What are the key principles of clean code architecture?")
    )

    print("📝 First request - should write to cache...")

    # First request with caching enabled
    results1 = await client.process_prompts_async(
        [conv1], tools=tools, cache="system_and_tools", show_progress=False
    )

    result1 = results1[0]
    assert result1 is not None, "First request failed"
    assert not result1.is_error, f"First request error: {result1.error_message}"
    assert result1.usage is not None, "No usage data in first result"

    print("✅ First request completed:")
    print(f"   Input tokens: {result1.usage.input_tokens}")
    print(f"   Output tokens: {result1.usage.output_tokens}")
    print(f"   Cache write tokens: {result1.usage.cache_write_tokens}")
    print(f"   Cache read tokens: {result1.usage.cache_read_tokens}")

    # Verify cache write occurred
    assert result1.usage.has_cache_write, "First request should have written to cache"
    assert (
        result1.usage.cache_write_tokens and result1.usage.cache_write_tokens > 0
    ), "Should have cache write tokens"

    # Create follow-up conversation (continuing the conversation)
    conv2 = (
        Conversation.system(system_msg)
        .add(Message.user("What are the key principles of clean code architecture?"))
        .add(Message.ai(result1.completion or "Here are the key principles..."))
        .add(
            Message.user(
                "Can you give me specific examples of how to implement these principles in Python?"
            )
        )
    )

    print("🔄 Second request - should read from cache...")

    # Second request with same caching setup
    results2 = await client.process_prompts_async(
        [conv2], tools=tools, cache="system_and_tools", show_progress=False
    )

    result2 = results2[0]
    assert result2 is not None, "Second request failed"
    assert not result2.is_error, f"Second request error: {result2.error_message}"
    assert result2.usage is not None, "No usage data in second result"

    print("✅ Second request completed:")
    print(f"   Input tokens: {result2.usage.input_tokens}")
    print(f"   Output tokens: {result2.usage.output_tokens}")
    print(f"   Cache write tokens: {result2.usage.cache_write_tokens}")
    print(f"   Cache read tokens: {result2.usage.cache_read_tokens}")

    # Verify cache read occurred
    assert result2.usage.has_cache_hit, "Second request should have read from cache"
    assert (
        result2.usage.cache_read_tokens and result2.usage.cache_read_tokens > 0
    ), "Should have cache read tokens"

    # Verify the cache read tokens approximately match the cache write tokens from first request
    # (allowing for small differences due to additional context)
    cache_write_first = result1.usage.cache_write_tokens or 0
    cache_read_second = result2.usage.cache_read_tokens or 0

    print("📊 Cache efficiency:")
    print(f"   Tokens written to cache: {cache_write_first}")
    print(f"   Tokens read from cache: {cache_read_second}")
    print(f"   Cache hit ratio: {cache_read_second/cache_write_first:.2%}")

    # The cache read should be significant (at least 80% of what was written)
    assert (
        cache_read_second >= cache_write_first * 0.8
    ), f"Cache read tokens ({cache_read_second}) should be close to cache write tokens ({cache_write_first})"

    print("🎉 Real caching integration test passed!")
    print(
        f"   Successfully cached {cache_write_first} tokens and retrieved {cache_read_second} tokens"
    )


if __name__ == "__main__":
    asyncio.run(test_real_caching_integration())
