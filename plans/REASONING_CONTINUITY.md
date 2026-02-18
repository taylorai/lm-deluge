lm-deluge: Code Review
Medium: Reasoning items are never emitted after log round-trip because Conversation.to_openai_responses() now only emits Thinking.raw_payload items. Thinking.id is serialized, but raw_payload is not, so tool-call followups constructed from logs will omit reasoning items and lose Responses API reasoning continuity.

Note: This is intended, since we were emitting faulty reasoning items (reasoning summaries) as if they were real, which led to 400s. In the future, consider checking if it has an rs_ id or a reasoning_ id, and only emit the rs_ ones, instead of relying on presence/absence of `raw_payload`.
