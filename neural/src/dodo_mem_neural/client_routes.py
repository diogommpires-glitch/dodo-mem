"""Keep client conversation structure intact while selecting genuine user queries."""

import re

from .contracts import CLIENTS, MEMORY_INSTRUCTIONS

CLINE_RESULT = re.compile(
    r"^\s*(?:\[SYSTEM\]|\[[^\n]+\]\s*Result:|<tool_result(?:\s|>)|<environment_details>)",
    re.IGNORECASE,
)
JAN_TITLE = (
    "Summarize the following conversation into a concise title of at most 10 words. "
    "Capture the overall topic, not just the latest turn. Output the title only, no quotes, "
    "no explanation.\n\nConversation:\n"
)


def text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    return ""


def auxiliary_request(messages, client):
    return (
        client == "jan"
        and len(messages) == 1
        and messages[0].get("role") == "user"
        and text_content(messages[0].get("content")).startswith(JAN_TITLE)
    )


def latest_user_text(messages: list[dict], client: str) -> str | None:
    if client not in CLIENTS:
        raise ValueError(f"unsupported client adapter: {client}")
    if auxiliary_request(messages, client):
        return None
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = text_content(message.get("content"))
        if client == "cline":
            if CLINE_RESULT.match(text):
                continue
            task = re.search(
                r"<(task|user_message|user_input)(?:\s[^>]*)?>(.*?)</\1>", text, re.DOTALL
            )
            if task:
                text = task[2]
            text = re.sub(
                r"<environment_details>.*?</environment_details>", "", text, flags=re.DOTALL
            )
        if text.strip():
            return text.strip()[:2000]
    return None


def inject_memory(messages: list[dict], memory_text: str, client: str) -> list[dict]:
    if client not in CLIENTS:
        raise ValueError(f"unsupported client adapter: {client}")
    if not memory_text:
        return list(messages)
    # Adding a separate system message preserves assistant reasoning and tool-call/result pairs.
    position = 0
    while position < len(messages) and messages[position].get("role") == "system":
        position += 1
    content = memory_text
    if "untrusted" not in content:
        content = MEMORY_INSTRUCTIONS + content
    return [*messages[:position], {"role": "system", "content": content}, *messages[position:]]
