import hashlib
import importlib.util
import json
from pathlib import Path


def configure_module():
    path = Path(__file__).parents[1] / "scripts/configure_clients.py"
    spec = importlib.util.spec_from_file_location("configure_clients", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_preserves_installed_profiles_and_separates_shared_credentials(tmp_path):
    module = configure_module()
    home, appdata = tmp_path / "home", tmp_path / "appdata"
    local = home / "local-ai"
    module.write_json(
        local / "models.json",
        {
            "coder": {
                "alias": "local-coder",
                "file": "models/coder.gguf",
                "port": 8098,
                "janProvider": "Local Coder",
            }
        },
    )
    state_path = local / "coder/cline-data/globalState.json"
    original = {
        "openAiBaseUrl": "http://127.0.0.1:8098/v1",
        "actModeOpenAiModelId": "local-coder",
        "planModeOpenAiModelId": "local-coder",
        "actModeReasoningEffort": "high",
        "unrelated": {"keep": True},
    }
    module.write_json(state_path, original)
    old_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    jan_path = appdata / "Jan/data/settings.json"
    module.write_json(
        jan_path,
        {
            "model-provider": json.dumps(
                {
                    "version": 0,
                    "state": {
                        "providers": [
                            {"provider": "Local Coder", "models": [{"id": "local-coder"}]}
                        ],
                        "selectedProvider": "Local Coder",
                        "selectedModel": {"id": "local-coder"},
                    },
                }
            )
        },
    )
    preset = local / "cline-plan-act/models.ini"
    preset.parent.mkdir(parents=True)
    preset.write_text("version = 1\n[*]\nload-on-startup = false\n", encoding="utf-8")
    manifest = module.stage(home, appdata)
    assert manifest["enabled"] is False
    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == old_hash
    assert len(manifest["files"]) == 2
    row = next(row for row in manifest["files"] if row["target"] == str(state_path))
    staged = module.read_json(row["staged"])
    assert staged["openAiBaseUrl"] == module.PROXY_URL
    assert staged["actModeOpenAiModelId"] == "coder"
    assert staged["actModeReasoningEffort"] == "high"
    assert staged["unrelated"] == original["unrelated"]
    assert hashlib.sha256(Path(row["backup"]).read_bytes()).hexdigest() == old_hash
    jan_row = next(row for row in manifest["files"] if row["target"] == str(jan_path))
    jan = json.loads(module.read_json(jan_row["staged"])["model-provider"])
    assert jan["state"]["selectedModel"]["id"] == "coder"
    routes = module.read_json(local / "dodo-mem-neural/routes.json")
    assert routes["routes"][0]["model_path"] == str(home / "models/coder.gguf")
    assert routes["routes"][1]["id"] == "paired-coder"
    assert all(row["scope"] == "shared" for row in routes["credentials"] if row["client"] != "dodo")
    assert len({row["token_sha256"] for row in routes["credentials"]}) == 5
    assert all("upstream_token" not in row for row in routes["routes"])
    assert module.connection("cline", "coder", "old-url", "old-key", "local-coder", home=home) == (
        "old-url",
        "old-key",
        "local-coder",
    )


def test_opencode_model_pins_follow_same_underlying_model_with_reasoning_unchanged():
    module = configure_module()
    model = {"alias": "local-coder"}
    profile = {
        "model": "local/local-coder",
        "small_model": "local-summary/local-coder",
        "agent": {"summary": {"model": "local-summary/local-coder"}},
        "provider": {
            name: {
                "options": {"baseURL": "old", "apiKey": "old"},
                "models": {
                    "local-coder": {
                        "reasoning": True,
                        "options": {"chat_template_kwargs": {"preserve_thinking": True}},
                    }
                },
            }
            for name in ("local", "local-summary")
        },
    }
    result = module.opencode_profile(profile, {"coder": model}, "secret")
    assert result["model"] == "local/coder"
    assert result["small_model"] == "local-summary/coder"
    assert result["agent"]["summary"]["model"] == "local-summary/coder"
    assert (
        result["provider"]["local"]["models"]["coder"]
        == profile["provider"]["local"]["models"]["local-coder"]
    )
    assert profile["provider"]["local"]["options"]["baseURL"] == "old"
