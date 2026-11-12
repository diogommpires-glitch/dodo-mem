"""The stable hosted route for the owner's reviewed personal profile."""

PROFILE_KEY = "personalization/approved-profile"
PROFILE_PATH = f"{PROFILE_KEY}.md"
WITHDRAWN_PROFILE = "<!-- DODO PROFILE WITHDRAWN -->"

ROUTING_INSTRUCTION = (
    "When the user says “contextualize with mem” or “contextualise with mem”, read Dodo Mem's "
    "get_core, then call get_note with personalization/approved-profile.md on this request. "
    "Apply the relevant approved preferences alongside the current request and project rules. "
    "If either call fails or the profile is unavailable, say so. Use Ask Dodo on this laptop "
    "for private context, or wait for explicit approval of an exact private excerpt packet."
)
