# Dodo Mem

Source-only implementation of a Markdown memory gateway, a native Windows desktop workspace and a local neural retrieval service, with evidence provenance, review gates and agent handoffs.

The gateway is at the repository root. `desktop/` contains the Windows workspace and `neural/` contains the retrieval, indexing and model-routing implementation. Each component has a dependency lock and synthetic offline tests. The repository contains no memory vault, personal profile, accepted notes, source documents, databases, backups, logs, runtime state, credentials, live configuration, model weights or memory-derived documentation.

The original test suite includes regressions derived from a personal memory archive. Those tests, their fixture loader and their archive are excluded. A new fixture declares a synthetic storage policy inside each temporary test directory. Disk and available-memory observations are synthetic in the unit suite; production resource floors and admission guards remain unchanged. Separate tests reject one byte below each real floor and admit at the exact floor. The included tests also cover complete table rows and units, typed physical quantities, scope checks, source hash verification and bounded evidence locations.

GitHub Actions runs this dataset-free unit suite on Windows and Ubuntu, with pinned actions, Python 3.14.4, uv 0.13.6 and the frozen dependency lock. Gitleaks 8.30.1 scans the checked Git tree with redacted findings. CI does not qualify the actual machine's capacity or a configured memory service.

Desktop and neural checks use separate Windows Python 3.13 environments. They exercise temporary catalogues, synthetic model inputs, in-process transports and disposable local test processes. They do not run a configured service, access a personal library, download model weights, train a model or qualify NPU hardware. See [COMPONENTS.md](COMPONENTS.md) for the explicit configuration boundaries and verification commands.

## Verify the source snapshot

Use Python 3.11 or later and uv:

```powershell
uv sync --locked
uv run --locked pytest tests
```

This repository is an unconfigured code snapshot and does not reproduce an installed deployment. State paths and library sources require explicit configuration. Missing personal profiles select no sources, and automatic whole-home, drive, chat-database and host-extension discovery is disabled. Client profiles, credentials, model artifacts and operational configuration must be supplied separately. The personal notification integration is omitted, including its MCP tool, and the gateway offline adapter requires an explicit runtime path. Windows and WSL deployment and real hardware operation have not been qualified here.

Original project code is source-available under the PolyForm Noncommercial License 1.0.0. See `LICENSE` and `LICENSING.md`. Commercial use is not granted by this project licence; third-party dependencies retain their own terms. Earlier grants for older versions are unaffected.
