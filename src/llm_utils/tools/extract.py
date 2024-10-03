import json
from PIL import Image as PILImage
from ..image import Image
from ..prompt import Prompt
import asyncio
from ..client import LLMClient
from typing import Optional, Any
from ..util.json import load_json


async def extract_async(
    inputs: list[str | PILImage.Image],
    schema: Any,
    client: LLMClient,
    document_name: Optional[str] = None,
    object_name: Optional[str] = None,
    show_progress: bool = True,
    return_prompts: bool = False
):
    if hasattr(schema, "model_json_schema"):
        schema_dict = schema.model_json_schema()
    elif isinstance(schema, dict):
        schema_dict = schema
    else:
        raise ValueError("schema must be a pydantic model or a dict.")

    # warn if json_mode is not True
    for sp in client.sampling_params:
        if sp.json_mode is False:
            print("Warning: json_mode is False for one or more sampling params. You may get invalid output.")
            break
    # check_schema(schema_dict) -- figure out later
    if document_name is None:
        document_name = "text"
    if object_name is None:
        object_name = ""
    else:
        object_name += " "

    text_only_prompt = (
        f"Given the following {document_name}, extract the {object_name}information "
        + "from it according to the following JSON schema:\n\n```json\n"
        + json.dumps(schema_dict, indent=2)
        + "```\n\nHere is the {document_name}:\n\n```\n{<<__REPLACE_WITH_TEXT__>>}\n```"
        + "Return the extracted information as JSON, no explanation required. "
        + f"If the {document_name} seems to be totally irrelevant to the schema (not just incomplete), you may return a JSON object "
        + "like `{\"error\": \"The document is not relevant to the schema.\"}`."
    )

    image_only_prompt = (
        f"Given the attached {document_name} image, extract the {object_name}information "
        + "from it according to the following JSON schema:\n\n```json\n"
        + json.dumps(schema_dict, indent=2)
        + "Return the extracted information as JSON, no explanation required. "
        + f"If the {document_name} seems to be totally irrelevant to the schema (not just incomplete), you may return a JSON object "
        + "like `{\"error\": \"The document is not relevant to the schema.\"}`."
    )

    prompts = []
    for input in inputs:
        if isinstance(input, str):
            prompts.append(text_only_prompt.replace("{<<__REPLACE_WITH_TEXT__>>}", input))
        elif isinstance(input, PILImage.Image):
            prompts.append(Prompt(text=image_only_prompt, image=Image(input)))
        else:
                raise ValueError("inputs must be a list of strings or PIL images.")

    if return_prompts:
        return prompts

    resps = await client.process_prompts_async(prompts, show_progress=show_progress)
    completions = [load_json(resp.completion) for resp in resps]

    return completions

def extract(
    inputs: list[str | PILImage.Image],
    schema: Any,
    client: LLMClient,
    document_name: Optional[str] = None,
    object_name: Optional[str] = None,
    show_progress: bool = True,
    return_prompts: bool = False
):
    return asyncio.run(extract_async(inputs, schema, client, document_name, object_name, show_progress, return_prompts))
