# Visual asset sources

The public source edition omits the undocumented animal image and icon. It uses
the procedural drawer and archive mark supplied with this component. The
original geometry and example plugin packs, built by
`tools/build_plugin_examples.py`, fall under the root project's non-commercial
terms. The notices below and the icon-directory licence retain their upstream
terms. Plugin ZIPs contain no separate replacement licence grant.

The archive drawer uses original geometry from [tools/render_drawer.py](tools/render_drawer.py).
The application reads the rendered PNGs and selected SVGs from its asset directory.

| Upstream release | Commit resolved from tag | Downloaded source ZIP SHA-256 | License notice |
|---|---|---|---|
| [Pygfx 0.17.0](https://github.com/pygfx/pygfx/releases/tag/v0.17.0) | `ed8e491b92ecc5739810898123c8c162a855bd7c` | `e7d6bb71f982908b439989fcffd9e49d5c7e33f2839a888e0fb32d2fc01362cd` | [BSD-2-Clause](tools/licenses/pygfx.txt) |
| [RenderCanvas 2.7.1](https://github.com/pygfx/rendercanvas/releases/tag/v2.7.1) | `f44ae725c79a7bec8df387be5b4f7fdc6f22a553` | `a8f8dabc7fc634c2661d1bf9ed3af20a7b032964c996e7b6abf6049ef7e21c6e` | [BSD-2-Clause](tools/licenses/rendercanvas.txt) |
| [wgpu-py 0.32.0](https://github.com/pygfx/wgpu-py/releases/tag/v0.32.0) | `9d59b27fab7a15c92cd1fc147912b889f08e93d3` | `dd5c7d0e8a6c543b1824aa6260bd948de41de183781ab418fa6bdc060bcc4d35` | [BSD-2-Clause](tools/licenses/wgpu-py.txt) |
| [Phosphor Core 2.0.8](https://github.com/phosphor-icons/core/releases/tag/v2.0.8) | `d42782b2abe747d904b971ccab48b182a1455f86` | `74a92a78bdcb969a0b6399fa0f3506878bbf9c7711395e3aeb26ea2343d1f3ef` | [MIT](src/dodo_memory_desktop/assets/icons/LICENSE) |

The ZIP hashes record the exact tag archives inspected from GitHub codeload.
The six regular-weight SVGs are copied unchanged from Phosphor's
`assets/regular` directory. Python renderer packages are installed only for
asset builds through the locked `visuals` group.

From the project directory, regenerate the artwork with an integrated or CPU
D3D12 adapter:

```powershell
uv run --locked --group visuals python tools/render_drawer.py
```

The result is a 640 × 440 still and twelve equally sized transparent PNG
frames. Play the frames in order at 36 ms per frame and return to the still.
