import json
from dataclasses import dataclass
import tiktoken
from .models import APIModel
from .prompt import Prompt
from typing import Union
from .image import Image
import xxhash

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

class Prompt:
    """
    A prompt contains a user message, optionally an image,
    optionally a system message. For now, not worrying about
    multi-turn conversations.
    """
    def __init__(self, text: Union[str, list[dict]], image: Image = None):
        self.image = image
        if isinstance(text, str):
            self.user_message = text
            self.system_message = None
        elif isinstance(text, list):
            if len(text) > 2:
                raise ValueError("Prompt can only have 2 messages.")
            elif len(text) == 1:
                if text[0]["role"] != "user":
                    raise ValueError("Must have a user message.")
                self.user_message = text[0]["content"]
                self.system_message = None
            else:
                if text[0]["role"] == "system" and text[1]["role"] == "user":
                    self.system_message = text[0]["content"]
                    self.user_message = text[1]["content"]
                else:
                    raise ValueError("First message must be system, second must be user.")
        else:
            raise ValueError("Prompt must be a string or a list of dictionaries.")
        
    @property
    def fingerprint(self):
        """
        A unique identifier for the prompt.
        """
        content = {
            "user_message": self.user_message,
            "system_message": self.system_message,
            "image": None if self.image is None else self.image.fingerprint
        }
        hasher = xxhash.xxh64()
        hasher.update(json.dumps(content).encode())
        return hasher.hexdigest()
    
    def count_tokens(self, max_new_tokens: int = 0, image_tokens: int = 0):
        text = self.user_message
        if self.system_message is not None:
            text = self.system_message + " " + text
        tokens = tokenizer.encode(text)
        num_tokens = 10 + len(tokens)

        if self.image is not None:
            num_tokens += image_tokens

        return num_tokens + max_new_tokens
    
    def dry_run(
        self,
        model_name: str,
        max_new_tokens: int,
    ):
        model_obj = APIModel.from_registry(model_name)
        if model_obj.api_spec == "openai":
            image_tokens = 85 
        elif model_obj.api_spec == "anthropic":
            image_tokens = 1_200
        input_tokens = self.count_tokens(0, image_tokens)
        output_tokens = max_new_tokens
        
        min_cost = model_obj.input_cost * input_tokens / 1e6 
        max_cost = min_cost + model_obj.output_cost * output_tokens / 1e6

        return input_tokens, output_tokens, min_cost, max_cost

    def to_openai(self):
        """
        Convert the prompt to a format that can be sent to the
        OpenAI API.
        """
        messages = []
        if self.system_message is not None:
            messages.append({
                "role": "system",
                "content": self.system_message
            })

        if self.image is not None:
            messages.append({
                "role": "user",
                "content": [
                    self.image.to_openai_input(),
                    {
                        "type": "text",
                        "text": self.user_message
                    }
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": self.user_message
            })
        
        return messages
    
    def to_cohere(self):
        # {
        #     "role": "USER" if message["role"] == "user" else "CHATBOT",
        #     "message": message["content"]
        # }
        if self.image is not None:
            raise ValueError("Cohere does not support images.")
        # for multi-turn, we'd fill this in
        chat_history = []
        return self.system_message, chat_history, self.user_message
    
    def to_gemini(self):
        system_instruction = None
        contents = []
        if self.system_message is not None:
            system_instruction = self.system_message

        if self.image is not None:
            contents.append({
                "role": "user",
                "parts": [
                    self.image.to_gemini_input(),
                    {"text": self.user_message}
                ]
            })
        else:
            contents.append({"role": "user", "parts": [{"text": self.user_message}]})

        return system_instruction, contents
    
    def to_anthropic(self):
        """
        Convert the prompt to a format that can be sent to the
        Anthropic API.
        """
        system_message = None
        messages = []
        if self.system_message is not None:
            system_message = self.system_message

        if self.image is not None:
            messages.append({
                "role": "user",
                "content": [
                    self.image.to_anthropic_input(),
                    {
                        "type": "text",
                        "content": self.user_message
                    }
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": self.user_message
            })
        
        return system_message, messages
    
    def to_log(self):
        return {
            "user_message": self.user_message,
            "system_message": self.system_message,
            "image": None if self.image is None else f"<Image ({self.image.num_pixels} pixels)>"
        }
    
    def from_log(self, log):
        self.user_message = log["user_message"]
        self.system_message = log["system_message"]
        if log["image"] is not None:
            print("WARNING: Image cannot be reconstructed from log.")
        return self