"""
Pydantic models used for request validation and response serialization.

These schemas define the public API contract of the LLM service.
Validation is performed automatically by FastAPI before the request
reaches the service layer.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List

class GenerateRequest(BaseModel):
    """
    Request schema for text generation.

    Represents a prompt-based generation request sent to the LLM.
    """
    model_config = ConfigDict(strict=True)

    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=2000,
        description="Input text prompt used to generate new text."
    )
    
    max_tokens: int = Field(
        50, 
        ge=1, 
        le=200,
        description="Maximum number of tokens to generate."
    )


class GenerateResponse(BaseModel):
    """
    Response schema for generated text.

    Contains the text produced by the language model.
    """
    text: str = Field(
        ...,
        description="Generated text output from the language model."
    )


class EncodeRequest(BaseModel):
    """
    Request schema for text encoding.

    Accepts raw text input and converts it into token IDs using the tokenizer.
    """

    model_config = ConfigDict(strict=True)
    
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=2000,
        description="Input text to be tokenized."
    )

class EncodeResponse(BaseModel):
    """
    Response schema containing token IDs.

    Returned after successful text tokenization.
    """
    tokens: List[int] = Field(
        ...,
        description="List of token IDs representing the encoded text."
    )


class DecodeRequest(BaseModel):
    """
    Request schema for decoding token IDs back into text.

    Accepts a list of integer token IDs and converts them back to text.
    """
    model_config = ConfigDict(strict=True)
    tokens: List[int] = Field(
        ..., 
        min_length=1,  
        max_length=4096,
        description="List of token IDs to be decoded into text."
    )


class DecodeResponse(BaseModel):
    """
    Response schema for decoded text.

    Contains the human-readable text reconstructed from token IDs.
    """
    text: str = Field(
        ...,
        description="Decoded text reconstructed from the provided token IDs."
    )