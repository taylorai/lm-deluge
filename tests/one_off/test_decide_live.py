"""Live smoke test for decision models. Requires the provider's API key, e.g.
TYPESAFE_API_KEY=... python tests/one_off/test_decide_live.py [model]"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge.decide import Choice, Noul, Score, decide, decide_parallel_async


async def main(model: str):
    resp = await decide(
        "Help! My payouts have been failing for 3 days and nobody answers support.",
        {
            "is_urgent": Noul("Does this convey urgency?"),
            "department": Choice(
                "Which team should handle this?",
                {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            ),
            "frustration": Score(
                "How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]
            ),
        },
        model=model,
    )
    assert not resp.is_error, f"{resp.status_code}: {resp.error_message}"
    print(resp.model, resp.input_tokens, resp.cost)
    for key, answer in resp.answers.items():
        print(f"  {key}: {answer}")
    assert set(resp.answers) == {"is_urgent", "department", "frustration"}
    print("PASSED: single decide")

    # Parallel, structured (JSON) state, raw-dict questions mixed with typed ones.
    tickets = [
        {"subject": "Refund request", "body": "I was charged twice for my plan."},
        {"subject": "500 errors", "body": "Your API returns 500 on every call."},
        {"subject": "Enterprise pricing", "body": "What does a 500-seat plan cost?"},
    ]
    results = await decide_parallel_async(
        tickets,
        {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            },
            "is_bug": Noul("This describes a software defect."),
        },
        model=model,
    )
    routed = []
    for r in results:
        assert not r.is_error, f"{r.status_code}: {r.error_message}"
        routed.append(r.answers["department"].choice)  # type: ignore
        print(f"  {r.state['subject']}: {r.answers}")
    assert routed == ["billing", "technical", "sales"], routed
    assert results[1].answers["is_bug"].noul > 0.5  # type: ignore
    print("PASSED: parallel decide")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "jev"))
