# heartbeat runs a model continuously, once every N
# minutes, letting it check if there's anything to do,
# doing whatever it wants, and then going back to sleep.
import asyncio

from lm_deluge import Conversation

SYSTEM_PROMPT = (
    "You are in a harness that runs on a loop, waking you "
    "up every 10 minutes. Each time you wake up, you can "
    "decide if there's anything you need to do, based on "
    "the actions available to you. You might decide nothing "
    "needs to be done, and you can just go back to sleep. "
    "If there are things that need done, you use the tools "
    "available to you to do them, then when you're done, "
    "you go back to sleep."
)

USER_MESSAGE = (
    "Your one job every time you wake up is to adjust "
    "the Phillips Hue lights. You have a skill to help you "
)


async def heartbeat():
    _conv = Conversation().system(SYSTEM_PROMPT)
    while True:
        await asyncio.sleep(600)  # Sleep for 10 minutes
        # Your code here to check for tasks and perform them
