"""One chat-format and completion boundary contract for SFT and TAB-PO."""
from __future__ import annotations


def messages(query: str, answer: str | None = None, system_prompt: str = "") -> list[dict]:
    result = [{"role": "system", "content": system_prompt}] if system_prompt else []
    result.append({"role": "user", "content": query})
    if answer is not None:
        result.append({"role": "assistant", "content": answer})
    return result


def completion_encoding(tokenizer, query: str, answer: str, max_length: int,
                        system_prompt: str = "") -> tuple[str, dict, list[int]]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for reliable completion-only masking")
    prefix = tokenizer.apply_chat_template(messages(query, system_prompt=system_prompt),
                                          tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages(query, answer, system_prompt),
                                        tokenize=False, add_generation_prompt=False)
    if not answer or not full.startswith(prefix + answer):
        raise ValueError("Chat template must preserve the prompt prefix and assistant content")
    enc = tokenizer(full, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    if len(enc["input_ids"]) > max_length:
        raise ValueError(f"Complete chat requires {len(enc['input_ids'])} tokens; max_length={max_length}. "
                         "Increase the limit; truncating schema instructions or JSON changes the task.")
    boundary = len(prefix)
    # The completion includes the template's assistant end-of-turn token.
    labels = [tid if start >= boundary and end > start else -100
              for tid, (start, end) in zip(enc["input_ids"], enc["offset_mapping"])]
    if not any(x != -100 for x in labels[1:]):
        raise ValueError("No answer tokens survive completion masking")
    return full, enc, labels
