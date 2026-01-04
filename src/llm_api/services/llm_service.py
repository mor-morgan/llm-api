"""
LLM service layer.

This module encapsulates all interactions with the underlying language model
and tokenizer. It is intentionally isolated from the API layer to keep business
logic decoupled from HTTP concerns.

The service is designed to be instantiated once at application startup and
reused across requests (e.g., as a singleton in the API layer).
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from llm_api.exceptions.llm_exceptions import (
    ModelLoadError,
    TokenizationError,
    GenerationError,
)
import logging

logger = logging.getLogger("llm_service")


class LLMService:
    """
    High-level service that wraps model/tokenizer operations.

    Responsibilities:
    - Load a Hugging Face causal language model and its tokenizer.
    - Provide convenience methods for text generation, encoding, and decoding.
    - Translate low-level library failures into domain-specific exceptions.

    Notes:
    - This class is stateful (holds the loaded model/tokenizer).
    - The methods are synchronous and intended for server-side usage.
    """

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the service by loading the tokenizer and model.

        Args:
            model_name: Hugging Face model identifier (e.g., "gpt2").

        Raises:
            ModelLoadError: If the tokenizer/model cannot be loaded
                (e.g., network/cache/weights issues).
        """
        logger.info("Loading model: %s", model_name)
        self.model_name = model_name
        try: 
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            # Fail fast if the model cannot be loaded (IO / weights / cache Error)
            raise ModelLoadError(f"Failed to load model '{model_name}'") from e


    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Generate a text continuation for the given prompt.

        Args:
            prompt: Input prompt text to condition generation on.
            max_tokens: Maximum number of new tokens to generate.

        Returns:
            A decoded string containing the generated continuation.

        Raises:
            GenerationError: If tokenization, model generation, or decoding fails.

        Notes:
            - Uses torch.no_grad() to avoid gradient tracking.
            - Uses sampling (do_sample=True), so outputs are non-deterministic
              unless you set a random seed externally.
        """
        logger.info("Generate called (max_tokens=%s)", max_tokens)#prompt might be sensitive
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise GenerationError("Text generation failed") from e    
        

    def encode(self, text: str) -> list[int]:
        """
        Convert input text into a list of token IDs.

        Args:
            text: Input text to tokenize.

        Returns:
            A list of integer token IDs.

        Raises:
            TokenizationError: If the tokenizer fails to encode the input.

        Notes:
            Special tokens are not added (add_special_tokens=False) to keep the
            returned list "clean" and consistent for round-tripping with decode.
        """
        logger.info("Encode called (text=%s)", text)
        try:
            # for clean token list without special tokens
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            raise TokenizationError("Failed to encode text") from e


    def decode(self, tokens: list[int]) -> str:
        """
        Convert a list of token IDs back into text.

        Args:
            tokens: List of token IDs produced by encode or compatible sources.

        Returns:
            Decoded text.

        Raises:
            TokenizationError: If the tokenizer fails to decode the tokens.

        Notes:
            Skips special tokens (skip_special_tokens=True) to produce a
            human-readable string.
        """
        logger.info("Decode called (tokens=%s)", tokens)
        try:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as e:
            raise TokenizationError("Failed to decode tokens") from e