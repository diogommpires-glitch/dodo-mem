# Desktop and neural source

These components expose the implementation, synthetic regression suites and locked dependencies. They do not include a ready-made personal deployment. The gateway remains a separately installed package at the repository root; the neural lock declares both local package dependencies explicitly.

## Configuration boundary

Set `DODO_MEM_RUNTIME_ROOT` to an absolute local directory reserved for this installation before importing or launching the desktop or neural components. The public code uses its `gateway`, `desktop` and `neural` children. It never falls back to a previous installed checkout. Parent traversal, network paths and redirected configured roots are rejected.

`DODO_MEM_LIBRARY_ROOTS` is a JSON array of explicitly selected local source roots, with at most 32 entries. Its default is an empty array. Personal context files must be inside those selected roots. A missing `personal-profile.json` selects no personal sources. Optional local model routes belong in the desktop state directory's `models.json`; a missing registry provides no model routes.

`DODO_MEM_CHAT_ROOTS` selects supported chat-history directories, and `DODO_MEM_CLINE_DATABASES` selects exact Cline catalogue paths. Both default to empty arrays. Supplying these settings is a deliberate local data-access decision. No chat database or history directory is discovered from the user's home by default. Host extension discovery likewise requires an explicit host list passed to its API.

The neural cache stays inside the selected neural state's `bulk` directory. Existing resource floors remain 5 GiB available RAM and 100 GiB free disk, with a 20 GiB cache quota. Unit tests use synthetic resource observations where stated; they do not establish that the current machine satisfies those floors. Approval, scope, source-hash, path, sandbox and resource guards remain active.

Do not point verification commands at an installed vault, runtime state or personal library. Actual services additionally need owner-supplied credentials, route/selection configuration, authorised sources and separately obtained model artifacts. This release supplies none of them. Model/device qualification, client-profile installation and operational migration are outside the offline checks below.

## Isolated checks

Use Windows and Python 3.13 for the native desktop and neural suites. From the repository root, install each component's frozen environment:

```powershell
uv sync --project desktop --locked --managed-python
uv sync --project neural --locked --managed-python
```

Choose a disposable directory outside the source tree, then configure only that directory for the tests. For example, using a new temporary directory:

```powershell
$testRoot = Join-Path $env:TEMP ([guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $testRoot | Out-Null
$env:DODO_MEM_RUNTIME_ROOT = Join-Path $testRoot 'runtime'
$env:DODO_MEM_LIBRARY_ROOTS = ConvertTo-Json -Compress -InputObject @($testRoot)
$env:DODO_MEM_CHAT_ROOTS = '[]'
$env:DODO_MEM_CLINE_DATABASES = '[]'
$env:QT_QPA_PLATFORM = 'offscreen'
Push-Location desktop
uv run --locked ruff check src tests tools
uv run --locked ruff format --check src tests tools
uv run --locked python -m pytest tests -q --basetemp "$testRoot/desktop-tests"
Pop-Location
Push-Location neural
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked python -m pytest tests -q --basetemp "$testRoot/neural-tests"
Pop-Location
```

The suites construct their own source records and temporary catalogues. Native process tests create and stop only their disposable test children on temporary loopback ports. They do not register a real scheduled task, start the installed service, contact a hosted model or download weights.

## Assets and terms

The desktop uses the original procedural archive drawer and icon, with the generator under `desktop/tools/render_drawer.py`. The undocumented animal photograph/icon is omitted. External icon notices and renderer dependency notices remain under the desktop component. The example plugin ZIPs contain only reviewed manifests and small example Wasm modules; their build source is `desktop/tools/build_plugin_examples.py`.

Original nested source, generated original artwork and example packs use the root PolyForm Noncommercial terms. Separate component copies of the same licence accompany built packages. Third-party notices and dependency licences retain their original terms, including every upstream copyright notice. No model or dataset is relicensed or redistributed. Earlier grants remain unaffected. See [LICENSING.md](LICENSING.md), [desktop/THIRD_PARTY.md](desktop/THIRD_PARTY.md) and [neural/THIRD_PARTY.md](neural/THIRD_PARTY.md).
