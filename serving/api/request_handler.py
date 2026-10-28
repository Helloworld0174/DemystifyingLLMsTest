"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from utils.logging_utils import get_logger

logger = get_logger("request_handler")


class RequestHandler:
    """
    Handles endpoint request validation, inference invocation, and response formatting.
    """

    def __init__(self, engine: Optional[Any] = None, tokenizer: Optional[Any] = None):
        self.engine = engine
        self.tokenizer = tokenizer

    def handle_completion(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process OpenAI-compatible /v1/completions request.

        Args:
            request_body: Dictionary containing 'prompt', optional 'max_tokens', 'temperature', etc.

        Returns:
            Formatted response dictionary.
        """
        prompt = request_body.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        max_tokens = request_body.get("max_tokens", 16)
        temperature = request_body.get("temperature", 0.8)
        if not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("temperature must be non-negative")
        top_k = request_body.get("top_k")
        top_p = request_body.get("top_p")
        if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
            raise ValueError("top_k must be a non-negative integer")
        if top_p is not None and (not isinstance(top_p, (int, float)) or not 0 <= top_p <= 1):
            raise ValueError("top_p must be in [0, 1]")
        if self.engine is None:
            raise RuntimeError("No inference engine is configured")

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = [ord(c) for c in prompt]

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generation_options = {
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            }
            if top_k is not None:
                generation_options["top_k"] = top_k
            if top_p is not None:
                generation_options["top_p"] = top_p
            generated_tokens = self.engine.generate_tokens(**generation_options)
        elif self.engine is not None and hasattr(self.engine, "generate"):
            generation_options = {
                "tokenizer": self.tokenizer,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            }
            if top_k is not None:
                generation_options["top_k"] = top_k
            if top_p is not None:
                generation_options["top_p"] = top_p
            raw_gen = self.engine.generate(prompt, **generation_options)
            generated_tokens = raw_gen
        else:
            raise RuntimeError("The configured inference engine cannot generate completions")

        if isinstance(generated_tokens, str):
            response_text = generated_tokens[len(prompt):] if generated_tokens.startswith(prompt) else generated_tokens
            completion_tokens = (
                max(0, len(self.tokenizer.encode(response_text)) - 1)
                if self.tokenizer is not None and hasattr(self.tokenizer, "encode")
                else len(response_text)
            )
        else:
            completion = generated_tokens[len(prompt_tokens):]
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                response_text = self.tokenizer.decode(completion)
            else:
                response_text = "".join(chr(token) for token in completion if 0 <= token < 0x110000)
            completion_tokens = len(completion)

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "broken-transformer",
            "text": response_text,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": (
                        "stop" if not isinstance(generated_tokens, str) and completion
                        and completion[-1] in getattr(getattr(self.engine, "inference_config", None), "stop_token_ids", [])
                        else "length"
                    )
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_tokens) + completion_tokens
            }
        }

    def handle_health(self) -> Dict[str, Any]:
        """Process /health endpoint check."""
        return {
            "status": "ok",
            "service": "broken-transformer-server",
            "timestamp": time.time()
        }

    def handle_models(self) -> Dict[str, Any]:
        """Process /v1/models endpoint check."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "broken-transformer",
                    "object": "model",
                    "owned_by": "openai-dsg-iitr",
                    "ready": True
                }
            ]
        }
