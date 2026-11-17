"""Stage local memory client connections without changing any installed profile."""

import argparse
import configparser
import copy
import hashlib
import json
import re
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path

from dodo_memory_desktop.configuration import runtime_path

PROXY_URL = "http://127.0.0.1:8771/v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def backup_file(path, directory):
    path, directory = Path(path), Path(directory)
    if not path.exists():
        return {"target": str(path), "before_sha256": None, "backup": None}
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    backup = directory / (hashlib.sha256(str(path).encode()).hexdigest()[:16] + "-" + path.name)
    if not backup.exists():
        shutil.copy2(path, backup)
    elif hashlib.sha256(backup.read_bytes()).hexdigest() != digest:
        raise ValueError(f"backup already exists for another revision: {path}")
    return {"target": str(path), "before_sha256": digest, "backup": str(backup)}


def connection(client, model, default_url, default_key, default_model, *, home=None):
    state = Path(home) / "local-ai/dodo-mem-neural" if home is not None else runtime_path("neural")
    active = state / "clients.active.json"
    if not active.exists() or read_json(active).get("enabled") is not True:
        return default_url, default_key, default_model
    routes = read_json(state / "routes.json")
    allowed = next(row for row in routes["credentials"] if row["client"] == client)
    if model not in allowed["routes"]:
        raise ValueError(f"memory route is not authorised for {client}: {model}")
    token = (state / f"{client}.token").read_text(encoding="utf-8").strip()
    if hashlib.sha256(token.encode()).hexdigest() != allowed["token_sha256"]:
        raise ValueError(f"memory credential changed for {client}")
    return PROXY_URL, token, model


def opencode_profile(value, models, token):
    value = copy.deepcopy(value)
    aliases = {model["alias"]: key for key, model in models.items()}
    replacements = {}
    for name in ("local", "local-summary"):
        provider = value["provider"][name]
        provider["options"].update(baseURL=PROXY_URL, apiKey=token)
        entries = {}
        for alias, entry in provider["models"].items():
            route = aliases.get(alias, alias if alias in models else None)
            if route is None:
                raise ValueError(f"OpenCode local model has no registered memory route: {alias}")
            entries[route] = entry
            replacements[f"{name}/{alias}"] = f"{name}/{route}"
        provider["models"] = entries
    for key in ("model", "small_model"):
        value[key] = replacements.get(value[key], value[key])
    for agent in value.get("agent", {}).values():
        if "model" in agent:
            agent["model"] = replacements.get(agent["model"], agent["model"])
    return value


def _cline_state(value, aliases, token):
    value = copy.deepcopy(value)
    value["openAiBaseUrl"] = PROXY_URL
    for mode in ("plan", "act"):
        for suffix in ("ModeOpenAiModelId", "ModeApiModelId"):
            key = mode + suffix
            if value.get(key) in aliases:
                value[key] = aliases[value[key]]
    return value


def stage(home, appdata):
    home, appdata = Path(home), Path(appdata)
    local = home / "local-ai"
    state = local / "dodo-mem-neural"
    state.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    staging = state / "staged" / stamp
    models = read_json(local / "models.json")
    tokens = {}
    for client in ("dodo", "jan", "cline", "opencode", "shared"):
        path = state / ("broker.token" if client == "dodo" else f"{client}.token")
        if not path.exists():
            path.write_text(secrets.token_urlsafe(36) + "\n", encoding="utf-8")
        tokens[client] = path.read_text(encoding="utf-8").strip()
        if len(tokens[client]) < 32:
            raise ValueError(f"local memory token must contain at least 32 characters: {client}")
    routes = []
    for key, model in models.items():
        relative = Path(model["file"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.casefold() != ".gguf"
        ):
            raise ValueError(f"model weight must be a home-relative GGUF: {key}")
        output = 2048 if key.startswith("qwen4b-") else 4096
        common = {
            "alias": model["alias"],
            "model_path": str(home / relative),
            "max_output_tokens": output,
        }
        routes.append(
            {"id": key, "url": f"http://127.0.0.1:{model['port']}", "raw_router": False, **common}
        )
        routes.append(
            {
                "id": "paired-" + key,
                "url": "http://127.0.0.1:8110",
                "raw_router": True,
                "upstream_token_file": str(local / "cline-plan-act/api-key.txt"),
                **common,
            }
        )
    preset = configparser.ConfigParser(interpolation=None)
    preset_path = local / "cline-plan-act/models.ini"
    if preset_path.exists():
        preset.read_string("[global]\n" + preset_path.read_text(encoding="utf-8-sig"))
    if preset.has_section("local-abiray-orca"):
        routes.append(
            {
                "id": "paired-orca",
                "alias": "local-abiray-orca",
                "url": "http://127.0.0.1:8110",
                "raw_router": True,
                "model_path": preset["local-abiray-orca"]["model"],
                "max_output_tokens": 4096,
                "upstream_token_file": str(local / "cline-plan-act/api-key.txt"),
            }
        )
    route_ids = [row["id"] for row in routes]
    credentials = [
        {
            "token_sha256": hashlib.sha256(tokens[client].encode()).hexdigest(),
            "client": client,
            "scope": "full_private" if client == "dodo" else "shared",
            "routes": route_ids,
        }
        for client in ("dodo", "jan", "cline", "opencode")
    ]
    credentials.append(
        {
            "token_sha256": hashlib.sha256(tokens["shared"].encode()).hexdigest(),
            "client": "dodo",
            "scope": "shared",
            "routes": [],
        }
    )
    write_json(state / "routes.json", {"routes": routes, "credentials": credentials})
    files = []

    def candidate(path, value, *, text=False):
        record = backup_file(path, staging / "backups")
        target = staging / "files" / f"{len(files):03d}-{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if text:
            target.write_text(value, encoding="utf-8")
        else:
            write_json(target, value)
        files.append(
            {
                **record,
                "staged": str(target),
                "after_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )

    jan_path = appdata / "Jan/data/settings.json"
    if jan_path.exists():
        jan = read_json(jan_path)
        providers = json.loads(jan["model-provider"])
        names = {model["janProvider"]: key for key, model in models.items()}
        for provider in providers["state"]["providers"]:
            if provider.get("provider") not in names:
                continue
            key = names[provider["provider"]]
            provider.update(base_url=PROXY_URL, api_key=tokens["jan"])
            for entry in provider.get("models", []):
                for field in ("name", "id", "model"):
                    entry[field] = key
        selected = providers["state"].get("selectedProvider")
        if selected in names:
            for field in ("name", "id", "model"):
                providers["state"]["selectedModel"][field] = names[selected]
        jan["model-provider"] = json.dumps(providers, separators=(",", ":"))
        candidate(jan_path, jan)

    aliases = {model["alias"]: key for key, model in models.items()}
    paired = {alias: "paired-" + key for alias, key in aliases.items()}
    paired["local-abiray-orca"] = "paired-orca"
    profiles = [(local / key / "cline-data", aliases) for key in models]
    profiles.extend([(local / "cline-plan-act/cline-data", paired), (home / ".cline/data", paired)])
    for profile, mapping in profiles:
        path = profile / "globalState.json"
        if path.exists():
            candidate(path, _cline_state(read_json(path), mapping, tokens["cline"]))
        path = profile / "secrets.json"
        if path.exists():
            candidate(path, {**read_json(path), "openAiApiKey": tokens["cline"]})
        path = profile / "settings/models.json"
        if path.exists():
            value = read_json(path)
            provider = value.get("providers", {}).get("openai-compatible")
            if provider:
                provider["provider"]["baseUrl"] = PROXY_URL
                default = provider["provider"].get("defaultModelId")
                if default in mapping:
                    provider["provider"]["defaultModelId"] = mapping[default]
                provider["models"] = {
                    mapping.get(key, key): entry for key, entry in provider["models"].items()
                }
                candidate(path, value)
        path = profile / "settings/providers.json"
        if path.exists():
            value = read_json(path)
            provider = value.get("providers", {}).get("openai-compatible")
            if provider:
                settings = provider["settings"]
                settings.update(baseUrl=PROXY_URL, apiKey=tokens["cline"])
                if settings.get("model") in mapping:
                    settings["model"] = mapping[settings["model"]]
                candidate(path, value)

    memory = home / "dodo-shared-memory"
    pinned = {}
    for name in ("opencode-memory.json", "opencode-coder-memory.json"):
        path = memory / "profiles" / name
        if path.exists():
            candidate(path, opencode_profile(read_json(path), models, tokens["opencode"]))
            pinned[name] = files[-1]["after_sha256"]
    launcher = memory / "scripts/start_opencode_memory.py"
    if launcher.exists() and len(pinned) == 2:
        source = launcher.read_text(encoding="utf-8")
        for constant, name in (
            ("PROFILE_SHA256", "opencode-memory.json"),
            ("CODER_PROFILE_SHA256", "opencode-coder-memory.json"),
        ):
            source, count = re.subn(
                rf'(?m)^{constant} = "[a-f0-9]+"$', f'{constant} = "{pinned[name]}"', source
            )
            if count != 1:
                raise ValueError(f"OpenCode launcher pin was not found exactly once: {constant}")
        marker = "    if actual != expected:\n"
        if source.count(marker) != 1:
            raise ValueError("OpenCode launcher expected-profile check was not found exactly once")
        script = str(Path(__file__).resolve())
        patch = f"    import runpy\n    memory_config = runpy.run_path({script!r})\n    registry = memory_config['read_json'](Path.home() / 'local-ai/models.json')\n    memory_token = (Path.home() / 'local-ai/dodo-mem-neural/opencode.token').read_text(encoding='utf-8').strip()\n    expected = memory_config['opencode_profile'](expected, registry, memory_token)\n"
        source = source.replace(marker, patch + marker)
        candidate(launcher, source, text=True)
    manifest = {
        "enabled": False,
        "created_at": stamp,
        "routes": len(routes),
        "files": files,
        "activation_requires": "runtime, current-scope privacy, real-client adapter and cancellation qualification",
    }
    write_json(staging / "manifest.json", manifest)
    write_json(
        state / "staged.json", {"enabled": False, "manifest": str(staging / "manifest.json")}
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument(
        "--appdata",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    manifest = stage(args.home, args.appdata)
    print(
        f"Staged {manifest['routes']} routes and {len(manifest['files'])} profile files; installed profiles remain unchanged."
    )


if __name__ == "__main__":
    main()
