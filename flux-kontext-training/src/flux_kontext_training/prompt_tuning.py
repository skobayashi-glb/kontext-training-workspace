from __future__ import annotations

from typing import Any, Protocol

from .transformer_flux2_pt_supported import Flux2Transformer2DModel


class TokenizerWithAddedTokens(Protocol):
    def add_tokens(self, new_tokens: list[str], special_tokens: bool = False) -> int: ...

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def __len__(self) -> int: ...


class TextEncoderWithResizeTokenEmbeddings(Protocol):
    def resize_token_embeddings(self, new_num_tokens: int) -> Any: ...


def add_prompt_tuning_tokens(
    tokenizer: TokenizerWithAddedTokens,
    transformer: Flux2Transformer2DModel,
    num_tokens: int,
    text_encoder: TextEncoderWithResizeTokenEmbeddings | None = None,
    init_std: float = 0.02,
) -> list[str]:
    tokens = transformer.get_prompt_tuning_tokens()
    if len(tokens) != num_tokens:
        tokens = [f"<flux2-pt-{index}>" for index in range(num_tokens)]
    added_count = tokenizer.add_tokens(tokens, special_tokens=True)
    if text_encoder is not None and added_count > 0:
        text_encoder.resize_token_embeddings(len(tokenizer))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    transformer.initialize_prompt_tuning(
        num_tokens=num_tokens,
        token_ids=token_ids,
        init_std=init_std,
    )
    return tokens


def prepend_prompt_tuning_tokens(prompt: str, prompt_tuning_tokens: list[str]) -> str:
    if not prompt_tuning_tokens:
        return prompt
    return " ".join([*prompt_tuning_tokens, prompt])
