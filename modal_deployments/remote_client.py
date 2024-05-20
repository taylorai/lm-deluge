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
         Secret.from_name("my-googlecloud-secret")
    ],
    concurrency_limit=1,
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
        if os.getenv("SERVICE_ACCOUNT_JSON", None) is not None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/service_account.json"
            json.dump(json.loads(os.getenv("SERVICE_ACCOUNT_JSON")), open("/service_account.json", "w"))
    
        print("Processing prompts...")
        result: list[APIResponse] = self.client.process_prompts_sync(prompts)
        now = datetime.datetime.now().strftime('%H-%M-%S-%m-%d-%Y')
        filename = f"/outputs/results-{now}.jsonl"
        with open(filename, "w") as f:
            for i, r in enumerate(result):
                result_json = r.to_dict()
                f.write(json.dumps(result_json) + "\n")
                if i % 10_000 == 0:
                    vol.commit()
                yield result_json    
        vol.commit()
        print("Done!")