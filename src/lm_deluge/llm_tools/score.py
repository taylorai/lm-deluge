from ..client import LLMClient, APIResponse
from ..util.logprobs import extract_prob

# def extract_prob_yes(logprobs: list[dict]):
#     """
#     Extract the log probability of the token "yes" from the logprobs object.
#     Since we can't rely on "yes" and "no" both being in the top_logprobs,
#     we do the following:
#     - if token is "yes", return p(yes)
#     - if token is "no", return 1 - p(no)
#     - otherwise, return 0.5
#     """
#     # use regexp to keep only alpha characters
#     top_token = logprobs[0]["token"].lower()
#     top_token = re.sub(r"[^a-z]", "", top_token)
#     if top_token == "yes":
#         return np.exp(logprobs[0]["logprob"])
#     elif top_token == "no":
#         return 1 - np.exp(logprobs[0]["logprob"])
#     else:
#         return 0.5


def score_llm(
    scoring_prompt_template: str,
    inputs: list[tuple | list | dict],  # to format the template
    scoring_model: LLMClient,
    return_probabilities: bool,
    yes_token: str = "yes",
) -> list[bool | None] | list[float | None]:
    if return_probabilities:
        if not hasattr(scoring_model, "logprobs") or not scoring_model.logprobs:
            raise ValueError(
                "return_probabilities=True requires scoring_model to have logprobs=True. "
                "you may need to upgrade lm_deluge to have access to this option."
            )

    if scoring_prompt_template is None:
        raise ValueError("scoring_prompt must be provided.")

    scoring_prompts = []
    for inp in inputs:
        if isinstance(inp, dict):
            scoring_prompt = scoring_prompt_template.format(**inp)
        elif isinstance(inp, tuple) or isinstance(inp, list):
            scoring_prompt = scoring_prompt_template.format(*inp)
        else:
            raise ValueError("inputs must be a list of tuples, lists, or dicts.")
        scoring_prompts.append(scoring_prompt)

    resps: list[APIResponse] = scoring_model.process_prompts_sync(  # pyright: ignore
        prompts=scoring_prompts,
        show_progress=False,
    )

    if return_probabilities:
        logprobs_list = [resp.logprobs for resp in resps]
        scores = [
            extract_prob(yes_token, logprobs, use_complement=True)
            if logprobs is not None
            else None
            for logprobs in logprobs_list
        ]
    else:
        completions = [resp.completion for resp in resps]
        scores = [
            yes_token.lower().strip() in c.lower() if c is not None else None
            for c in completions
        ]

    return scores  # p(yes) or bool yes/no
