import re
import numpy as np
from typing import TypedDict, Callable


class TopLogprob(TypedDict):
    token: str
    logprob: float
    bytes: list[int]


class LogprobEntry(TypedDict):
    token: str
    logprob: float
    bytes: list[int]
    top_logprobs: list[TopLogprob]


Logprobs = list[LogprobEntry]

## In our implementation of APIResponse, the 'logprobs' field contains
## just the 'content' field from the respons.choices[0].logprobs object.
# {
#   "id": "chatcmpl-A6izyp6wnlEv6SLAb0ehIwBqCDLyR",
#   "object": "chat.completion",
#   "created": 1726166306,
#   "model": "gpt-4o-mini-2024-07-18",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "A loop within loops,",
#         "refusal": null
#       },
#       "logprobs": {
#         "content": [
#           {
#             "token": "A",
#             "logprob": -1.0330456,
#             "bytes": [
#               65
#             ],
#             "top_logprobs": [
#               {
#                 "token": "A",
#                 "logprob": -1.0330456,
#                 "bytes": [
#                   65
#                 ]
#               },
#               {
#                 "token": "In",
#                 "logprob": -2.0330458,
#                 "bytes": [
#                   73,
#                   110
#                 ]
#               },
#               {
#                 "token": "Nested",
#                 "logprob": -2.0330458,
#                 "bytes": [
#                   78,
#                   101,
#                   115,
#                   116,
#                   101,
#                   100
#                 ]
#               },
#               {
#                 "token": "Function",
#                 "logprob": -2.7830458,
#                 "bytes": [
#                   70,
#                   117,
#                   110,
#                   99,
#                   116,
#                   105,
#                   111,
#                   110
#                 ]
#               },
#               {
#                 "token": "Layers",
#                 "logprob": -3.1580458,
#                 "bytes": [
#                   76,
#                   97,
#                   121,
#                   101,
#                   114,
#                   115
#                 ]
#               }
#             ]
#           },
#           {
#             "token": " loop",
#             "logprob": -2.909274,
#             "bytes": [
#               32,
#               108,
#               111,
#               111,
#               112
#             ],
#             "top_logprobs": [
#               {
#                 "token": " function",
#                 "logprob": -0.9092741,
#                 "bytes": [
#                   32,
#                   102,
#                   117,
#                   110,
#                   99,
#                   116,
#                   105,
#                   111,
#                   110
#                 ]
#               },
#               {
#                 "token": " call",
#                 "logprob": -1.0342741,
#                 "bytes": [
#                   32,
#                   99,
#                   97,
#                   108,
#                   108
#                 ]
#               },
#               {
#                 "token": " task",
#                 "logprob": -2.409274,
#                 "bytes": [
#                   32,
#                   116,
#                   97,
#                   115,
#                   107
#                 ]
#               },
#               {
#                 "token": " loop",
#                 "logprob": -2.909274,
#                 "bytes": [
#                   32,
#                   108,
#                   111,
#                   111,
#                   112
#                 ]
#               },
#               {
#                 "token": " problem",
#                 "logprob": -4.034274,
#                 "bytes": [
#                   32,
#                   112,
#                   114,
#                   111,
#                   98,
#                   108,
#                   101,
#                   109
#                 ]
#               }
#             ]
#           },
#           {
#             "token": " within",
#             "logprob": -0.09628018,
#             "bytes": [
#               32,
#               119,
#               105,
#               116,
#               104,
#               105,
#               110
#             ],
#             "top_logprobs": [
#               {
#                 "token": " within",
#                 "logprob": -0.09628018,
#                 "bytes": [
#                   32,
#                   119,
#                   105,
#                   116,
#                   104,
#                   105,
#                   110
#                 ]
#               },
#               {
#                 "token": " in",
#                 "logprob": -2.72128,
#                 "bytes": [
#                   32,
#                   105,
#                   110
#                 ]
#               },
#               {
#                 "token": " of",
#                 "logprob": -4.47128,
#                 "bytes": [
#                   32,
#                   111,
#                   102
#                 ]
#               },
#               {
#                 "token": " that",
#                 "logprob": -5.34628,
#                 "bytes": [
#                   32,
#                   116,
#                   104,
#                   97,
#                   116
#                 ]
#               },
#               {
#                 "token": " inside",
#                 "logprob": -5.59628,
#                 "bytes": [
#                   32,
#                   105,
#                   110,
#                   115,
#                   105,
#                   100,
#                   101
#                 ]
#               }
#             ]
#           },
#           {
#             "token": " loops",
#             "logprob": -0.12761699,
#             "bytes": [
#               32,
#               108,
#               111,
#               111,
#               112,
#               115
#             ],
#             "top_logprobs": [
#               {
#                 "token": " loops",
#                 "logprob": -0.12761699,
#                 "bytes": [
#                   32,
#                   108,
#                   111,
#                   111,
#                   112,
#                   115
#                 ]
#               },
#               {
#                 "token": " self",
#                 "logprob": -3.127617,
#                 "bytes": [
#                   32,
#                   115,
#                   101,
#                   108,
#                   102
#                 ]
#               },
#               {
#                 "token": " loop",
#                 "logprob": -3.627617,
#                 "bytes": [
#                   32,
#                   108,
#                   111,
#                   111,
#                   112
#                 ]
#               },
#               {
#                 "token": " calls",
#                 "logprob": -4.377617,
#                 "bytes": [
#                   32,
#                   99,
#                   97,
#                   108,
#                   108,
#                   115
#                 ]
#               },
#               {
#                 "token": " itself",
#                 "logprob": -4.877617,
#                 "bytes": [
#                   32,
#                   105,
#                   116,
#                   115,
#                   101,
#                   108,
#                   102
#                 ]
#               }
#             ]
#           },
#           {
#             "token": ",",
#             "logprob": -1.7432603e-6,
#             "bytes": [
#               44
#             ],
#             "top_logprobs": [
#               {
#                 "token": ",",
#                 "logprob": -1.7432603e-6,
#                 "bytes": [
#                   44
#                 ]
#               },
#               {
#                 "token": "  \n",
#                 "logprob": -13.875002,
#                 "bytes": [
#                   32,
#                   32,
#                   10
#                 ]
#               },
#               {
#                 "token": "—",
#                 "logprob": -14.750002,
#                 "bytes": [
#                   226,
#                   128,
#                   148
#                 ]
#               },
#               {
#                 "token": ",\n",
#                 "logprob": -15.000002,
#                 "bytes": [
#                   44,
#                   10
#                 ]
#               },
#               {
#                 "token": ";",
#                 "logprob": -17.375002,
#                 "bytes": [
#                   59
#                 ]
#               }
#             ]
#           }
#         ],
#         "refusal": null
#       },
#       "finish_reason": "length"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 28,
#     "completion_tokens": 5,
#     "total_tokens": 33
#   },
#   "system_fingerprint": "fp_483d39d857"
# }


def normalize_token(token: str):
    return re.sub(r"[^a-z]", "", token.lower())


def is_match(token1: str, token2: str):
    token1 = normalize_token(token1)
    token2 = normalize_token(token2)
    if token1 == token2:
        return True
    elif token1.startswith(token2):
        return True
    elif token2.startswith(token1):
        return True
    else:
        return False


def extract_prob(
    token: str,
    logprobs: Logprobs,
    use_top_logprobs: bool = False,
    normalize_top_logprobs: bool = True,  # if using top_logprobs, normalize by all the present tokens so they add up to 1
    use_complement: bool = False,  # if True, assume there's 2 choices, and return 1 - p if the top token doesn't match
    token_index: int = 0,  # get from the first token of the completion by default
    token_match_fn: Callable[[str, str], bool] | None = is_match,
):
    """
    Extract the probability of the token from the logprobs object of a single
    completion.
    """
    # ensure the token_index is valid
    if token_index >= len(logprobs):
        raise ValueError("token_index must be less than the length of logprobs.")
    entry: LogprobEntry = logprobs[token_index]
    # if using top_logprobs, ensure that at least one top_logprob is present
    if use_top_logprobs:
        if entry.get("top_logprobs", None) is None or len(entry["top_logprobs"]) == 0:
            raise ValueError(
                "top_logprobs must be present in logprobs to use top_logprobs=True."
            )
        top_tokens = [t["token"] for t in entry["top_logprobs"]]
        top_probs = [np.exp(t["logprob"]) for t in entry["top_logprobs"]]
        combined_prob = sum(
            [p for t, p in zip(top_tokens, top_probs) if is_match(t, token)]
        )

        if normalize_top_logprobs:
            # no point in using complement if normalizing; it will always be 0 if not present
            return combined_prob / sum(top_probs)
        elif combined_prob > 0:
            return combined_prob
        elif use_complement:
            return 1 - combined_prob
        else:
            return 0.0

    else:
        top_token = entry["token"]
        top_prob = np.exp(entry["logprob"])
        if is_match(top_token, token):
            return top_prob
        elif use_complement:
            return 1 - top_prob
        else:
            return 0.0
