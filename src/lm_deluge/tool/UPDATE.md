Current State of lm_deluge/tool/__init__.py

  What Each Component Does

  1. _python_type_to_json_schema_enhanced() (lines
  24-72)
  - Converts Python type hints to JSON Schema
  - Handles: primitives, Optional, Literal, list[T],
   dict[str, T]
  - Limitation: Doesn't handle Union, nested
  Pydantic models, TypedDict, Annotated[T,
  description], or complex types

  2. ToolParams class (lines 75-211)
  - Helper to construct tool parameter schemas
  - Multiple ways to create:
    - Direct dict: ToolParams({"city": str})
    - With extras: ToolParams({"op": (str, {"enum":
  [...]})})
    - from_pydantic(model) - extracts JSON schema
  from Pydantic model
    - from_typed_dict(td) - extracts schema from
  TypedDict
    - from_json_schema(props, required) - wraps
  existing JSON schema
  - Use case: Manual schema construction separate
  from functions
  - Problem: Rarely used directly—most people want
  Tool.from_function()

  3. Tool class (lines 259-823)
  - The core tool definition
  - Multiple creation methods:
    - Constructor: Tool(name=..., parameters=...,
  run=...)
    - Tool.from_function(fn): Introspects function →
   builds schema
    - Tool.from_pydantic(name, model): Pydantic
  model defines params (model IS the schema, not
  output)
    - Tool.from_typed_dict(name, td): TypedDict
  defines params
    - Tool.from_params(name, params): Uses
  ToolParams helper
    - Tool.from_mcp(server_name, ...): Loads tools
  from MCP server
    - Tool.from_mcp_config(config): Loads from
  Claude Desktop-style config
    - Tool.built_in(name, ...): For provider
  built-ins (web search, etc.)

  4. MCPServer class (lines 833-894)
  - Pass MCP servers directly to providers that
  support server-side MCP
  - Works with Anthropic MCP connector and OpenAI
  responses API

  ---
  Key Questions Answered

  "What does it mean to define a tool from a
  Pydantic model?"

  The Pydantic model defines the input parameters
  (schema), not the output. You still need to
  provide a run function. The model is essentially a
   typed way to define the JSON schema:

  class SearchQuery(BaseModel):
      query: str
      max_results: int = 10

  # The model defines WHAT parameters the tool
  accepts
  tool = Tool.from_pydantic("search", SearchQuery,
  run=my_search_fn)

  "Does function transformation work with
  TypedDict/Pydantic return types?"

  No. Looking at from_function() (lines 343-384), it
   only extracts:
  1. Function name
  2. Docstring (for description)
  3. Input parameter types → JSON schema
  4. Required list (from defaults)

  Return types are completely ignored. Compare to
  fastmcp which:
  - Extracts output schema from return type
  annotation
  - Supports Annotated[T, "description"] for
  parameter descriptions
  - Uses pydantic.TypeAdapter for robust schema
  generation

  ---
  Problems Identified

  1. _python_type_to_json_schema_enhanced is weak
    - Doesn't use Pydantic's TypeAdapter (the modern
   robust approach)
    - Missing: Union types, nested models, forward
  refs, Annotated metadata
  2. No docstring parsing for parameter descriptions
    - fastmcp parses docstrings AND supports
  Annotated[str, "description"]
    - Our code just uses the whole docstring as tool
   description
  3. Too many overlapping constructors
    - from_pydantic, from_typed_dict, from_params
  all do similar things
    - ToolParams class is rarely needed if
  from_function is robust
  4. No output schema support
    - fastmcp generates output schemas from return
  types
    - OpenAI's structured outputs feature needs this
  5. Strict mode handling is hacky
    - Multiple places with duplicated logic for
  OpenAI strict mode
    - Should centralize in one place

  ---
  Improvement Plan

  Phase 1: Strengthen from_function() (Core)

  1. Use Pydantic's TypeAdapter for schema
  generation (like fastmcp)
    - Handles all Python types correctly
    - Supports forward refs, Annotated, complex
  unions
  2. Support Annotated[Type, "description"] for
  parameter descriptions
  def search(query: Annotated[str, "Search query"],
  limit: int = 10): ...
  3. Optional: Parse docstrings for parameter
  descriptions (Google/Numpy style)
    - Lower priority—Annotated is cleaner

  Phase 2: Add Output Schema Support

  1. Extract return type and generate output schema
  2. Support Annotated[ReturnType, "description"] on
   return

  Phase 3: Simplify API

  1. Deprecate ToolParams class - absorbed into
  from_function
  2. Keep but simplify:
    - from_function() - primary way
    - from_pydantic() - for when you have a model
    - Direct constructor - escape hatch
  3. Remove or deprecate:
    - from_typed_dict() - TypedDict is awkward for
  this
    - from_params() - just use constructor

  Phase 4: Clean Up Internals

  1. Centralize strict mode schema transformation
  2. Extract common schema utilities to separate
  module
  3. Add better validation for edge cases

  ---
  What to Steal from Reference Implementations

  From fastmcp:
  - ParsedFunction class pattern (line 396-538 in
  tool.py)
  - get_cached_typeadapter() for robust schema
  generation
  - Annotated[T, "description"] → Annotated[T,
  Field(description=...)] conversion
  - Output schema inference from return types

  From openai-python:
  - to_strict_json_schema() for strict mode
  transformation
  - Clean $ref unraveling logic

  ---
  Want me to start implementing any of these phases?
