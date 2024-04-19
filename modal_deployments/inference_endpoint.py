# run gemma with modal
import os
from functools import partial
from modal import Image, Secret, Stub, enter, method, gpu

MODEL_DIR = "/model"

MODEL_OPTIONS = {
    "mistral-7b": {
        "name": "mistral-7b",
        "path": "mistralai/Mistral-7B-Instruct-v0.2",
    },
    "llama3-8b": {
        "name": "llama-8b",
        "path": "meta-llama/Llama-3-8b-chat-hf",
    },
    "gemma-7b": {
        "name": "gemma-7b",
        "path": "google/gemma-7b-it"
    },
    "nous-mistral-7b": {
        "name": "nous-mistral-7b",
        "path": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    }
}

GPU_OPTIONS = {
    "a100": gpu.A100(count=1, memory=80),
    "h100": gpu.H100(count=1, memory=80),
    "l4": gpu.L4(count=1)
}

model_name = os.environ.get("MODEL_NAME", "llama3-8b")
model_path = MODEL_OPTIONS[model_name]["path"]
gpu_name = os.environ.get("GPU", "a100").lower()
gpu_config = GPU_OPTIONS[gpu_name]

def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"Downloading {model_name} from HuggingFace...")
    snapshot_download(
        model_path,
        local_dir=MODEL_DIR,
        token=os.environ["HUGGINGFACE_TOKEN"],
        ignore_patterns=["*.pt", "*.gguf"],
    )
    move_cache()


image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
    )
    .pip_install(
        "vllm==0.3.2",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "torch==2.1.2",
    ).pip_install_private_repos(
        "github.com/taylorai/llm_utils@c78e7c2",
        secrets=[Secret.from_name("my-github-secret")],
        git_user="andersonbcdefg",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        partial(download_model_to_folder, model_path=model_path),
        secrets=[Secret.from_name("HF-SECRET")],
        timeout=60 * 20,
    )
)

stub = Stub(f'{model_name}-completions-{gpu_name}')

@stub.cls(
    image=image,
    gpu=gpu_config,
    secrets=[Secret.from_name("HF-SECRET")],
    timeout=60 * 60
)
class Model:
    @enter()
    def load(self):
        from vllm import LLM
        from transformers import AutoTokenizer

        # Load the model. Tip: Some models, like MPT, may require `trust_remote_code=true`.
        self.llm = LLM(MODEL_DIR, enforce_eager=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    @method()
    def generate(
        self, 
        prompts: list[list[dict]], 
        sampling_params: dict # should correspond to llm_utils sampling params spec
    ):
        import time
        from llm_utils.api_requests.base import APIResponse
        from llm_utils.sampling_params import SamplingParams
        sampling_params = SamplingParams(**sampling_params)
        vllm_sampling_params = sampling_params.to_vllm()

        tokenized = [
            self.tokenizer.apply_chat_template(p, add_generation_prompt=True) for p in prompts
        ]
        # any model can have JSON mode if you only dream!
        if sampling_params.json_mode:
            prompts = [p + "\n```json\n" for p in prompts]
            # TODO: figure out how to stop when the ``` is closed.

        result = self.llm.generate(prompts, vllm_sampling_params)

        responses = []
        for idx, output in enumerate(result):
            responses.append(APIResponse(
                model_internal=model_name + "-modal",
                system_prompt=None,
                messages=prompts[idx],
                sampling_params=sampling_params,
                status_code=200,
                is_error=False,
                error_message=None,
                completion=output.outputs[0].text,
                input_tokens=len(tokenized[idx]),
                output_tokens=len(output.outputs[0].token_ids),
                model_external=model_name,
                region=None,
                cost=None,
                finish_reason=None # TODO: get this
            ))
            
        return [
            res.to_dict() for res in responses
        ]