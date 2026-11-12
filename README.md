# Dodo Mem

Source-only snapshot of a local Markdown memory service with bounded retrieval, evidence provenance, review gates and agent handoffs.

This repository contains the Python package, dependency locks and four test modules that construct synthetic fixtures in temporary directories. It contains no memory vault, personal profile, accepted notes, source documents, databases, backups, logs, runtime state, credentials, live configuration or memory-derived documentation.

The original test suite includes regressions derived from a personal memory archive. Those tests, their fixture loader and their archive are excluded. A new fixture declares a synthetic storage policy inside each temporary test directory. Disk and available-memory observations are synthetic in the unit suite; production resource floors and admission guards remain unchanged. Separate tests reject one byte below each real floor and admit at the exact floor. The included tests also cover complete table rows and units, typed physical quantities, scope checks, source hash verification and bounded evidence locations.

GitHub Actions runs this dataset-free unit suite on Windows and Ubuntu, with pinned actions, Python 3.14.4, uv 0.13.6 and the frozen dependency lock. Gitleaks 8.30.1 scans the checked Git tree with redacted findings. CI does not qualify the actual machine's capacity or a configured memory service.

## Verify the source snapshot

Use Python 3.11 or later and uv:

```powershell
uv sync --locked
uv run --locked pytest tests
```

The installed service has not been changed. This repository is an unconfigured code snapshot and does not reproduce its deployment. Client profiles, launch scripts, owner configuration and operational instructions must be created separately from synthetic templates before use. The personal notification integration is omitted, including its MCP tool, and the offline adapter requires an explicit runtime path. Its Windows and WSL deployment has not been qualified here.

Original project code is source-available under the PolyForm Noncommercial License 1.0.0. See `LICENSE` and `LICENSING.md`. Commercial use is not granted by this project licence; third-party dependencies retain their own terms. Earlier grants for older versions are unaffected.
