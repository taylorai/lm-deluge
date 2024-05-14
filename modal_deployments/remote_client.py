# basically a proxy to use Modal to run llm_utils
# to avoid issues with local network
import os
import datetime
from modal import App, Image, Secret, Volume, method
from typing import Union

vol = Volume.from_name("llm_utils", create_if_missing=True)

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("PyYAML")
    .pip_install_private_repos(
        "github.com/taylorai/llm_utils@7f929cc",
        secrets=[Secret.from_name("my-github-secret")],
        git_user="andersonbcdefg",
    )
)

app = App("llm-utils")

@app.cls(
    image=image,
    volumes={"/outputs": vol},
    secrets=[
        Secret.from_name("OPENAI_API_KEY"),
        Secret.from_name("ANTHROPIC_API_KEY"),
        Secret.from_name("COHERE_API_KEY"),
    ],
    concurrency_limit=10,
    timeout=60 * 60 * 24
)
class ModalLLMClient:
    def __init__(self, **kwargs):
        from llm_utils import LLMClient
        self.client = LLMClient(**kwargs)

    @method()
    def process_prompts(self, prompts: Union[list[str], list[list[dict]]]):
        import json
        from llm_utils import APIResponse
        print("Processing prompts...")
        result: list[APIResponse] = self.client.process_prompts_sync(prompts)
        result_json = [r.to_dict() for r in result]
        # log result to volume in case it's lost.
        # file should start with HH-mm-ss-MM-DD-YYYY
        print("Writing results to file...")
        now = datetime.datetime.now().strftime('%H-%M-%S-%m-%d-%Y')
        filename = f"/outputs/results-{now}.json"
        with open(filename, "w") as f:
            f.write(json.dumps(result_json))
        vol.commit()
        print("Done!")
        return result_json