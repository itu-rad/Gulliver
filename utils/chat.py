"""Chat-template helpers that handle models with a thinking mode.

Thinking is left on by default; consumers strip the reasoning block with
`split_thinking`.
"""

from __future__ import annotations

from typing import Any

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Appended when the reasoning budget runs out before the model closes its
# block (follows Qwen's thinking-budget recipe).
THINK_FORCE_CLOSE = (
    "\n\nConsidering the limited time by the user, I have to give the "
    "solution based on the thinking directly now.\n" + THINK_CLOSE + "\n\n"
)


def apply_chat_template_safe(
    tokenizer,
    chat: list[dict[str, str]],
    *,
    enable_thinking: bool = True,
    **kwargs: Any,
) -> str:
    """Apply chat template with `enable_thinking` if the tokenizer accepts it.

    Falls back to the plain call for tokenizers that don't know the kwarg
    (raises TypeError on unexpected keyword).

    Args:
        tokenizer: HuggingFace-style tokenizer with `apply_chat_template`.
        chat: List of `{"role": ..., "content": ...}` dicts.
        enable_thinking: Whether to allow the model's thinking mode.
        **kwargs: Forwarded to `apply_chat_template` (e.g. `tokenize`,
            `add_generation_prompt`).

    Returns:
        The rendered prompt string.
    """
    try:
        return tokenizer.apply_chat_template(
            chat, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        # Tokenizer doesn't know `enable_thinking` (Llama, etc.).
        return tokenizer.apply_chat_template(chat, **kwargs)


def split_thinking(text: str) -> tuple[str, str, bool]:
    """Split a generation into (reasoning, answer, closed).

    `closed` is False when the model never emitted the closing tag: the whole
    text is then reasoning and the answer is empty. The opening tag is part of
    the prompt under Qwen's template, so the generation may start mid-block;
    an opening tag inside the generation is tolerated either way.
    """
    if THINK_CLOSE in text:
        reasoning, answer = text.split(THINK_CLOSE, 1)
        if THINK_OPEN in reasoning:
            reasoning = reasoning.split(THINK_OPEN, 1)[1]
        return reasoning.strip(), answer.strip(), True
    if THINK_OPEN in text:
        return text.split(THINK_OPEN, 1)[1].strip(), "", False
    return text.strip(), "", False
