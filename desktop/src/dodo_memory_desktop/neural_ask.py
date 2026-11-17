"""Opt-in rollout of authenticated local memory, with explicit memory-off requests."""

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .ask_evidence import AskError
from .configuration import runtime_path

STATE = runtime_path("neural")
PROXY = "http://127.0.0.1:8771"
BROKER = "http://127.0.0.1:8770"
_MEMORY_OFF_LOCK = threading.Lock()
_UNCONFIRMED_ROUTES = set()


def activated(home=STATE):
    try:
        selection = json.loads((home / "selection.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as error:
        raise AskError("The local memory rollout settings could not be read.") from error
    return selection.get("activated") is True


def _description(packet, question):
    sources = []
    for original in packet.get("sources", []):
        source = dict(original)
        source["relationship_path"] = [
            {
                "from": edge["source"],
                "to": edge["target"],
                "kind": edge["kind"],
                "evidence": edge.get("evidence"),
            }
            for edge in source.get("relationship_path", [])
        ]
        sources.append(source)
    coverage = packet.get("coverage", {})
    return {
        "question": question,
        "sources": sources,
        "connections": [],
        "coverage": {
            "index_generation": None,
            "index_scanning": False,
            "unavailable_evidence_reads": 0,
            "bounded_results": "budget_exhausted" in coverage.get("coverage_gaps", []),
            **coverage,
        },
    }


@dataclass(frozen=True)
class NeuralSnapshot:
    description: dict
    packet_id: str = ""
    route: str = ""
    token: str = field(default="", repr=False)
    transport: object = field(default=None, repr=False)

    @property
    def sources(self):
        return tuple(self.description["sources"])

    @property
    def connections(self):
        return tuple(self.description["connections"])

    def verify(self, _catalogue):
        if not self.packet_id:
            return
        try:
            with httpx.Client(transport=self.transport, trust_env=False, timeout=5) as client:
                response = client.post(
                    BROKER + "/validate",
                    json={"packet_id": self.packet_id, "model": self.route},
                    headers={"Authorization": f"Bearer {self.token}"},
                )
                response.raise_for_status()
                if response.json().get("valid") is not True:
                    raise AskError(
                        "The displayed evidence could not be revalidated. Run the question again."
                    )
        except (httpx.HTTPError, ValueError) as error:
            raise AskError(
                "The displayed evidence changed or expired. Run the question again."
            ) from error


def _messages(question, skill):
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise AskError("Ask a question containing 1 to 2000 characters.")
    instructions = "Answer the user's question. Cite relevant supplied memory evidence with its source numbers and report uncertainty."
    if skill:
        instructions += "\nUser-selected local guidance:\n" + json.dumps(
            {"name": skill[0], "guidance": skill[1]}, ensure_ascii=False
        )
    return [{"role": "system", "content": instructions}, {"role": "user", "content": question}]


def _ready(client, route):
    origin = urlsplit(route["url"])
    if (
        origin.scheme != "http"
        or origin.hostname not in {"127.0.0.1", "localhost", "::1"}
        or origin.username
        or origin.password
        or origin.path not in {"", "/"}
        or origin.query
        or origin.fragment
    ):
        raise AskError("A configured local model route is not a loopback origin.")
    headers = {}
    if route.get("upstream_token_file"):
        secret = Path(route["upstream_token_file"]).read_text(encoding="utf-8").strip()
        headers["Authorization"] = f"Bearer {secret}"
    params = {"model": route["alias"], "autoload": "false"} if route.get("raw_router") else None
    try:
        response = client.get(
            route["url"].rstrip("/") + "/props", headers=headers, params=params, timeout=2
        )
        response.raise_for_status()
        props = response.json()
        return (
            props.get("model_alias") == route["alias"]
            and isinstance(props.get("model_path"), str)
            and Path(props["model_path"]) == Path(route["model_path"])
        )
    except (httpx.HTTPError, ValueError):
        return False


def _answer(client, url, **kwargs):
    with client.stream("POST", url, **kwargs) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_bytes(chunk_size=8192):
            body.extend(chunk)
            if len(body) > 256 * 1024:
                raise AskError("The local model response exceeded the size limit.")
        headers = response.headers
    result = json.loads(body)
    text = result["choices"][0]["message"].get("content")
    if not isinstance(text, str) or not text.strip():
        raise AskError("The local model returned no visible answer.")
    return text, headers


def _confirm_idle(client, route):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            response = client.get(route["url"] + "/slots", timeout=1)
            response.raise_for_status()
            slots = response.json()
        except (httpx.HTTPError, ValueError):
            _UNCONFIRMED_ROUTES.add(route["url"])
            raise AskError(
                "The local backend idle state is unavailable. This route is blocked."
            ) from None
        if (
            isinstance(slots, list)
            and slots
            and all(isinstance(slot, dict) and slot.get("is_processing") is False for slot in slots)
        ):
            return
        time.sleep(0.1)
    _UNCONFIRMED_ROUTES.add(route["url"])
    raise AskError(
        "Local cancellation could not be confirmed within ten seconds. This route is blocked."
    )


def ask_neural(question, context, skill, models, *, home=STATE, transport=None, node_id=None):
    try:
        routes = json.loads((home / "routes.json").read_text(encoding="utf-8"))["routes"]
        token = (home / "broker.token").read_text(encoding="utf-8").strip()
        with httpx.Client(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(300, connect=2),
        ) as client:
            selected = None
            for model in models:
                candidates = [
                    route
                    for route in routes
                    if route["alias"] == model.alias
                    and model.file
                    and Path(route["model_path"]) == Path(model.file)
                ]
                candidates.sort(key=lambda route: bool(route.get("raw_router")))
                selected = next((route for route in candidates if _ready(client, route)), None)
                if selected:
                    break
            if selected is None:
                raise AskError("The selected local model is not already running and ready.")
            messages = _messages(question, skill)
            selected_context = {
                key: str(value)
                for key, value in {
                    "workspace": context.workspace,
                    "node_id": context.node_id,
                }.items()
                if value is not None
            }
            if selected_context:
                messages[0]["content"] += "\nLocal context selected by the user:\n" + json.dumps(
                    selected_context
                )
            answer, headers = _answer(
                client,
                PROXY + "/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": selected["id"],
                    "messages": messages,
                    "stream": False,
                    "max_tokens": 1200,
                    **({"dodo_context": {"node_id": node_id}} if node_id else {}),
                },
            )
            packet_id = headers.get("X-Dodo-Packet")
            if not packet_id:
                raise AskError("The local memory proxy did not identify its evidence packet.")
            packet = client.post(
                BROKER + "/packet",
                headers={"Authorization": f"Bearer {token}"},
                json={"packet_id": packet_id, "model": selected["id"]},
            )
            packet.raise_for_status()
            snapshot = NeuralSnapshot(
                _description(packet.json(), question), packet_id, selected["id"], token, transport
            )
            return answer, snapshot, selected["alias"]
    except (OSError, httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
        raise AskError(
            "Local memory is unavailable. Retry, or disable memory for this request."
        ) from error


def inspect_neural(question, *, home=STATE, transport=None, node_id=None):
    try:
        token = (home / "broker.token").read_text(encoding="utf-8").strip()
        with httpx.Client(transport=transport, trust_env=False, timeout=30) as client:
            response = client.post(
                BROKER + "/context",
                headers={"Authorization": f"Bearer {token}"},
                json={"query": question, **({"node_id": node_id} if node_id else {})},
            )
            response.raise_for_status()
            packet = response.json()
            return NeuralSnapshot(
                _description(packet, question),
                packet["packet_id"],
                "local-inspection",
                token,
                transport,
            )
    except (OSError, httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        raise AskError(
            "Local memory evidence is unavailable. Retry when the service is ready."
        ) from error


def ask_without_memory(question, skill, models, *, transport=None):
    for model in models:
        if not model.file:
            continue
        route = {
            "alias": model.alias,
            "url": f"http://127.0.0.1:{model.port}",
            "model_path": str(model.file),
        }
        try:
            with httpx.Client(
                transport=transport,
                trust_env=False,
                follow_redirects=False,
                timeout=httpx.Timeout(300, connect=2),
            ) as client:
                if not _ready(client, route):
                    continue
                with _MEMORY_OFF_LOCK:
                    if route["url"] in _UNCONFIRMED_ROUTES:
                        raise AskError(
                            "This local route has an unconfirmed cancelled request. Restart after confirming it is idle."
                        )
                    _confirm_idle(client, route)
                    try:
                        answer, _headers = _answer(
                            client,
                            route["url"] + "/v1/chat/completions",
                            json={
                                "model": model.alias,
                                "messages": _messages(question, skill),
                                "stream": False,
                                "max_tokens": 1200,
                            },
                        )
                    finally:
                        _confirm_idle(client, route)
                snapshot = NeuralSnapshot(
                    _description({"coverage": {"memory_enabled": False}}, question)
                )
                return answer, snapshot, model.alias
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
            raise AskError(
                "The selected local model request failed with memory disabled."
            ) from error
    raise AskError("The selected local model is not already running and ready.")
