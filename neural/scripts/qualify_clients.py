"""Validate offline message pairs; installed-client qualification is excluded."""


def tool_pairs_valid(messages):
    pending, seen = set(), set()
    for message in messages:
        if message.get("role") == "assistant":
            if pending:
                return False
            for call in message.get("tool_calls", []):
                identity = call.get("id")
                if not isinstance(identity, str) or not identity or identity in seen:
                    return False
                pending.add(identity)
                seen.add(identity)
        elif message.get("role") == "tool":
            identity = message.get("tool_call_id")
            if identity not in pending:
                return False
            pending.remove(identity)
    return not pending
