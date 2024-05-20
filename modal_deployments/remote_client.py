# basically a proxy to use Modal to run llm_utils
# to avoid issues with local network
import os
import datetime
from modal import App, Image, Secret, Volume, method
from typing import Union

vol = Volume.from_name("llm_utils", create_if_missing=True)

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "PyYAML",
        "pandas",
        "Faker"
    )
    .pip_install_private_repos(
        "github.com/taylorai/llm_utils@fa68230",
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
        import pandas as pd
        from llm_utils import APIResponse
        from faker import Faker
        fake = Faker()
        if os.getenv("SERVICE_ACCOUNT_JSON", None) is not None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/service_account.json"
            json.dump(json.loads(os.getenv("SERVICE_ACCOUNT_JSON")), open("/service_account.json", "w"))
    
        print("Processing prompts...")
        now = datetime.datetime.now().strftime('%H-%M-%S-%m-%d-%Y')
        fake_name = fake.name()
        filename = f"/outputs/{fake_name.replace(' ', '-')}results-{now}.jsonl".lower()
        result: list[APIResponse] = self.client.process_prompts_sync(prompts)
        result_jsons = [
            r.to_dict() for r in result
        ]
        df = pd.DataFrame.from_records(result_jsons)
        df.to_json(
            filename,
            orient="records",
            lines=True
        )
        vol.commit()
        print("Saved to " + filename)

        return result_jsons