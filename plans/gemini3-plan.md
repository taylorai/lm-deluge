Plan for Gemini 3 Integration

  1. Data Model Updates (prompt.py)

  Extend the Thinking class to support thought signatures:
  - Use "thought_signature: str | None", not raw_payload. raw_payload appears half-baked and probably not implemented correctly for OpenAI, let's not make a further mess.
  - Update Thinking.gemini() method to emit thought
  signatures when converting back to Gemini format.
  - Do the same thought_signature: str | None = None
  for the ToolCall class.

  2. Configuration Updates (config.py)

  For SamplingParams:
  - We don't need to add thinking_level, we already have reasoning_effort. This should work, it just needs to be translated to the right gemini parameter.
  - Add media_resolution: Literal["media_resolution_low",
  "media_resolution_medium", "media_resolution_high"] |
  None = None, ignored by all models except Gemini3

  3. Gemini Request Builder Updates
  (api_requests/gemini.py)
    - Map reasoning_effort to thinkingLevel in
  generationConfig (for gemini3 only).
  - Add support for media_resolution parameter (requires
  v1alpha API version)
  - Keep backward compatibility with Gemini 2.5 models
  using thinking_budget, choose path based on model name

  Update response parsing (handle_response):
  - Parse thoughtSignature from response parts
  - Signatures can appear in:
    - functionCall parts (for tool calls)
    - Regular text parts (final chunk in streaming, or
  non-streaming response)
  - Store signatures in the appropriate Part object:
    - For functionCall: store in
  ToolCall.thought_signature
    - For text/thinking: store in
  Thinking.thought_signature

  4. Conversation Serialization Updates (prompt.py)

  Update to_gemini() method (line 1471):
  - When converting messages back to Gemini format,
  preserve thought signatures
  - For function calls: include thoughtSignature field in
  the part if present
  - For multi-step function calling: ensure all signatures
   are preserved in order
  - For parallel function calls: only first call has
  signature (per Gemini 3 spec)

  Handle signature propagation:
  - When building conversation history for multi-turn
  calls:
    - Sequential function calls: include all accumulated
  signatures
    - User must send back signature with the function call
   in next turn
    - This is strictly validated for function calling (400
   error if missing)
  - For text/chat: signatures recommended but not strictly
   required

  Special case - Context Engineering:
  - When importing conversations from other models (e.g.,
  Gemini 2.5 → 3), use dummy signature:
    - "thoughtSignature":
  "context_engineering_is_the_way_to_go". Make this the default thinkingSignature anytime you're trying to serialize a conversation for Gemini3 and there's a missing thinking_signature; but also edit warnings.py and print a warning so the user knows we made this substitution.

  5. Model Registry Updates (models/google.py)

  Add Gemini 3 Pro model:
  "gemini-3-pro-preview": {
      "id": "gemini-3-pro-preview",
      "name": "gemini-3-pro-preview",
      "api_base":
  "https://generativelanguage.googleapis.com/v1alpha",
      "api_key_env_var": "GEMINI_API_KEY",
      "supports_json": True,
      "supports_logprobs": False,
      "api_spec": "gemini",
      "input_cost": 2.0,  # <200k tokens
      "cached_input_cost": 0.5,  # estimated (needs
  confirmation)
      "output_cost": 12.0,  # <200k tokens
      # Note: >200k tokens: $4/$18 - may need tiered
  pricing support
      "reasoning_model": True,
  }

  Since the endpoint can be changed by model, we'll use v1alpha for gemini3.

  6. Message/Part Serialization Updates

  Update to_log() and from_log() methods:
  - Save thought signatures when logging conversations
  - Restore thought signatures when rehydrating from logs
  - Add thought_signature to JSON serialization in:
    - Message.to_log() (line 344) for thinking parts
    - Message.from_log() (line 399) for thinking parts
    - Conversation.to_log() (line 1527) for thinking/tool
  parts
    - Conversation.from_log() (line 1574) for
  thinking/tool parts

  7. Testing Strategy

  Create new test files:
  - tests/models/test_gemini_3_thinking_level.py - Test
  thinking_level parameter
  - tests/models/test_gemini_3_thought_signatures.py -
  Test signature preservation
  - tests/models/test_gemini_3_function_calling.py - Test
  multi-step and parallel function calling with signatures
  - tests/models/test_gemini_3_media_resolution.py - Test
  media_resolution parameter

  Test cases needed:
  1. Basic Gemini 3 request with thinking_level=high
  2. Function calling with thought signature preservation
  3. Multi-step sequential function calls (accumulating
  signatures)
  4. Parallel function calls (only first has signature)
  5. Text/chat with optional signatures
  6. Migration from Gemini 2.5 (dummy signature)
  7. Context engineering scenario

  8. Backward Compatibility

  Maintain support for Gemini 2.5:
  - For 2.5 models, reasoning_effort still mapped to a budget instead of to thinking_level
  - Use version detection logic to choose correct
  parameter
  
  9. Documentation Updates

  Update docs to cover:
  - New Gemini 3 model and its capabilities
  - Thought signature handling (mostly automatic via
  library)
  - media_resolution parameter usage
  - Temperature recommendation (keep at 1.0 for Gemini 3)
  - Pricing (tiered based on token count)

  10. Open Questions/Decisions

  1. Tiered pricing: Just use the lower pricing for now.
  2. API version: v1alpha for Gemini 3 only
  3. Automatically preserve all signatures.
    - Provide validation warnings if signatures are
  missing
  4. Medium thinking level: just send it to the API and let it fail, that way it will work when they start supporting it.
