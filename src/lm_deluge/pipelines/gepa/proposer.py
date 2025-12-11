"""
Proposer for GEPA optimization.

The proposer analyzes a trajectory and proposes improvements to ONE component.
"""

from __future__ import annotations

from lm_deluge.prompt import Conversation

from lm_deluge.pipelines.gepa.core import Component, EvalResult, Proposal
from lm_deluge.pipelines.gepa.util import (
    extract_text_from_response,
    format_components_for_prompt,
    format_conversation_compact,
)


DEFAULT_PROPOSAL_PROMPT = """You are optimizing an AI system by improving its text configuration.

## The Trajectory

Below is a conversation showing what the AI did on a task:

<trajectory>
{trajectory}
</trajectory>

## Feedback

{feedback}

## Components

These are the text components that control the AI's behavior. You can modify ONE of them:

{components}
{meta_instructions}
## Your Task

1. Analyze the trajectory to understand what went wrong (or could be better)
2. Identify which component is most likely responsible
3. Propose a specific improvement to that ONE component

Think about:
- Did the AI misunderstand the task? (maybe the system prompt needs clarity)
- Did it use tools incorrectly? (maybe tool descriptions need improvement)
- Did it miss important information? (maybe instructions need to be more explicit)

## Response Format

Respond with:
COMPONENT: <name of the component to change>
REASONING: <1-2 sentences on why this change will help>
NEW_VALUE:
```
<the complete improved text for this component>
```
"""


def build_proposal_prompt(
    conversation: Conversation,
    feedback: str,
    components: dict[str, Component],
    current_values: dict[str, str],
    template: str | None = None,
    meta_instructions: str | None = None,
) -> str:
    """
    Build the prompt for the proposer LLM.

    Args:
        conversation: The trajectory to analyze
        feedback: Feedback on the result
        components: Component definitions (with descriptions)
        current_values: Current text values for each component
        template: Optional custom prompt template
        meta_instructions: Optional instructions to guide the proposer's behavior
                          (e.g., "Focus on general improvements, don't overfit to specific examples")

    Returns:
        Formatted prompt string
    """
    template = template or DEFAULT_PROPOSAL_PROMPT

    # Format trajectory
    trajectory_str = format_conversation_compact(conversation)

    # Format components
    descriptions = {name: comp.description for name, comp in components.items()}
    components_str = format_components_for_prompt(current_values, descriptions)

    # Format meta instructions
    if meta_instructions:
        meta_str = f"\n## Guidelines\n\n{meta_instructions}\n\n"
    else:
        meta_str = "\n"

    return template.format(
        trajectory=trajectory_str,
        feedback=feedback,
        components=components_str,
        meta_instructions=meta_str,
    )


def parse_proposal_response(
    response: str, valid_components: list[str]
) -> Proposal | None:
    """
    Parse the proposer's response to extract the proposal.

    Args:
        response: Raw LLM response
        valid_components: List of valid component names

    Returns:
        Proposal if parsing succeeded, None otherwise
    """
    # Find COMPONENT line
    component_name = None
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("COMPONENT:"):
            component_name = line.split(":", 1)[1].strip()
            break

    if not component_name:
        return None

    # Validate component name
    if component_name not in valid_components:
        # Try case-insensitive match
        for valid in valid_components:
            if valid.lower() == component_name.lower():
                component_name = valid
                break
        else:
            return None

    # Find REASONING line
    reasoning = ""
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
            break

    # Extract new value from code block
    new_value = extract_text_from_response(response)
    if not new_value:
        return None

    return Proposal(
        component_name=component_name,
        new_value=new_value,
        reasoning=reasoning,
    )


async def propose_improvement(
    proposer_client,  # LLMClient
    eval_result: EvalResult,
    components: dict[str, Component],
    current_values: dict[str, str],
    prompt_template: str | None = None,
    meta_instructions: str | None = None,
) -> Proposal | None:
    """
    Use an LLM to propose an improvement to one component.

    Args:
        proposer_client: LLMClient for generating proposals
        eval_result: The evaluation result containing trajectory and feedback
        components: Component definitions
        current_values: Current text values
        prompt_template: Optional custom prompt template
        meta_instructions: Optional guidelines to steer the proposer
                          (e.g., "Don't overfit to specific examples")

    Returns:
        Proposal if successful, None otherwise
    """
    # Build prompt
    prompt = build_proposal_prompt(
        conversation=eval_result.conversation,
        feedback=eval_result.feedback,
        components=components,
        current_values=current_values,
        template=prompt_template,
        meta_instructions=meta_instructions,
    )

    # Call LLM
    response = await proposer_client.start(prompt)
    if not response or not response.completion:
        return None

    response_text = response.completion

    # Parse response
    valid_components = list(components.keys())
    return parse_proposal_response(response_text, valid_components)


def propose_improvement_sync(
    proposer_client,  # LLMClient
    eval_result: EvalResult,
    components: dict[str, Component],
    current_values: dict[str, str],
    prompt_template: str | None = None,
) -> Proposal | None:
    """
    Synchronous version of propose_improvement.
    """
    # Build prompt
    prompt = build_proposal_prompt(
        conversation=eval_result.conversation,
        feedback=eval_result.feedback,
        components=components,
        current_values=current_values,
        template=prompt_template,
    )

    # Call LLM
    responses = proposer_client.process_prompts_sync([prompt], show_progress=False)
    if not responses or not responses[0].completion:
        return None

    response_text = responses[0].completion

    # Parse response
    valid_components = list(components.keys())
    return parse_proposal_response(response_text, valid_components)
