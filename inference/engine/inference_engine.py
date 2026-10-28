"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from utils.logging_utils import get_logger
from inference.sampling import Sampler
from inference.generation.kv_cache import KVCache

logger = get_logger("inference_engine")


class FullTransformerModel(nn.Module):
    """Fallback modular transformer wrapper combining embeddings, blocks, and lm_head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.n_layers
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead

        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        if config.tie_weights:
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def forward(self, input_ids: torch.Tensor, offset: int = 0, **kwargs) -> torch.Tensor:
        h = self.embeddings(input_ids, offset=offset)
        h = self.blocks(h, **kwargs)
        logits = self.lm_head(h)
        return logits


class InferenceEngine:
    """
    End-to-end inference engine managing model state, token generation loop,
    and sampling orchestration.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        model: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.model_config = model_config if model_config is not None else ModelConfig()

        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device) if device else torch.device("cpu")

        if model is not None:
            self.model = model
        else:
            self.model = FullTransformerModel(self.model_config)

    def _prepare_model(self) -> None:
        """
        Internal lifecycle helper to prepare model state before inference execution.
        """
        if self.model is not None and hasattr(self.model, "train"):
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            self.model.eval()

    def generate_tokens(
        self,
        prompt_tokens: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> List[int]:
        """
        Generate a list of token IDs autoregressively from input token sequence.
        """
        self._prepare_model()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_tokens <= 0:
            return list(prompt_tokens)
        if not prompt_tokens:
            raise ValueError("prompt_tokens must contain at least one token")
        if self.model is None or not callable(self.model):
            raise RuntimeError("No model is available for generation")
        if self.inference_config.beam_width > 1:
            from inference.generation.beam_search import BeamSearch
            input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            search = BeamSearch(
                beam_width=self.inference_config.beam_width,
                max_length=max_tokens,
                eos_token_id=(self.inference_config.stop_token_ids[0]
                              if self.inference_config.stop_token_ids else None),
            )
            with torch.no_grad():
                return search.search(self.model, input_tensor)[0].tolist()

        temp = temperature if temperature is not None else self.inference_config.temperature
        sampler = Sampler(
            temperature=temp,
            top_k=top_k if top_k is not None else self.inference_config.top_k,
            top_p=top_p if top_p is not None else self.inference_config.top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
        )
        tokens = list(prompt_tokens)
        kv_cache = None
        if self.inference_config.use_kv_cache and getattr(self.model, "num_layers", 0) > 0:
            kv_cache = KVCache(self.model.num_layers)

        # Autoregressive generation loop
        with torch.no_grad():
            for _ in range(max_tokens):
                if kv_cache is not None and kv_cache.get_seq_length() > 0:
                    model_tokens = tokens[-1:]
                    offset = len(tokens) - 1
                else:
                    model_tokens = tokens
                    offset = 0
                input_tensor = torch.tensor([model_tokens], dtype=torch.long, device=self.device)

                if self.model is None or not callable(self.model):
                    raise RuntimeError("No model is available for generation")
                if kv_cache is not None:
                    logits = self.model(input_tensor, kv_cache=kv_cache, offset=offset)
                else:
                    logits = self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token_logits = logits[:, -1, :]
                next_token = sampler.sample(next_token_logits, generated_tokens=tokens).item()

                tokens.append(int(next_token))

                if next_token in self.inference_config.stop_token_ids:
                    break

        return tokens

    def generate(
        self,
        prompt: str,
        tokenizer: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        End-to-end string-to-string text generation.
        """
        self._prepare_model()
        # Tokenize prompt
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = [ord(c) for c in prompt]

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)
        return "".join(chr(token) for token in output_ids if 0 <= token < 0x110000)
