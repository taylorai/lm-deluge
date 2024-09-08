import json
# from pydantic import BaseModel
from ..client import LLMClient
from typing import Optional, Any

def parse_json(text: Optional[str]):
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```json"):
        text = text.split("```json", 1)[1]
    text = text.rstrip("`")

    return json.loads(text)


def extract(
    texts: list[str],
    schema: Any,
    client: LLMClient,
    document_name: Optional[str] = None,
    object_name: Optional[str] = None,
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

    prompt = (
        f"Given the following {document_name}, extract the {object_name}information "
        + "from it according to the following JSON schema:\n\n```json\n"
        + json.dumps(schema_dict, indent=2)
        + "```\n\nHere is the {document_name}:\n\n```\n{<<__REPLACE_WITH_TEXT__>>}\n```"
        + "Return the extracted information as JSON, no explanation required."
    )

    prompts = [prompt.replace("{<<__REPLACE_WITH_TEXT__>>}", text) for text in texts]
    resps = client.process_prompts_sync(prompts)
    completions = [parse_json(resp.completion) for resp in resps]

    return completions
