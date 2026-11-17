"""Render the archive artwork with the optional visuals dependency group."""

from pathlib import Path

import pygfx as gfx
import wgpu
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QRadialGradient
from rendercanvas.offscreen import OffscreenRenderCanvas

ASSETS = Path(__file__).resolve().parents[1] / "src/dodo_memory_desktop/assets"
WIDTH, HEIGHT = 640, 440
FRAME_COUNT = 12


def box(parent, dimensions, position, material):
    mesh = gfx.Mesh(gfx.box_geometry(*dimensions), material)
    mesh.local.position = position
    parent.add(mesh)


def drawer():
    scene = gfx.Scene()
    ink = gfx.MeshPhongMaterial(color="#272f2c", shininess=15)
    rim = gfx.MeshPhongMaterial(color="#3f4c45", shininess=20)
    brass = gfx.MeshPhongMaterial(color="#aa8051", shininess=52)
    recess = gfx.MeshPhongMaterial(color="#111713", shininess=5)
    papers = ("#c5b7a0", "#e6d7bd", "#c7b798", "#e9dbc5", "#d7c6a8", "#b8a58a")

    box(scene, (4.9, 0.34, 3.48), (0, 0.25, 0), ink)
    for x in (-2.35, 2.35):
        box(scene, (0.22, 0.73, 3.58), (x, 0.54, -0.03), ink)
        box(scene, (0.23, 0.04, 3.58), (x, 0.91, -0.03), rim)
        box(scene, (0.035, 0.045, 2.95), (x * 0.923, 0.74, 0), brass)
        for z in (-1.28, 1.28):
            box(scene, (0.32, 0.11, 0.37), (x * 0.83, 0.04, z), recess)
    box(scene, (4.68, 0.72, 0.17), (0, 0.55, -1.74), ink)
    box(scene, (4.98, 0.78, 0.2), (0, 0.57, 1.76), ink)
    box(scene, (4.98, 0.038, 0.21), (0, 0.97, 1.76), rim)
    box(scene, (4.55, 0.026, 0.024), (0, 0.29, 1.87), brass)
    box(scene, (1.14, 0.32, 0.045), (0, 0.64, 1.89), brass)
    box(scene, (0.91, 0.205, 0.035), (0, 0.64, 1.91), recess)
    for x in (-0.34, 0.34):
        box(scene, (0.072, 0.105, 0.14), (x, 0.64, 1.965), brass)
    box(scene, (0.75, 0.09, 0.067), (0, 0.64, 2.026), brass)

    lifted = gfx.Group()
    for index, colour in enumerate(papers):
        divider = lifted if index == 2 else gfx.Group()
        divider.local.z = -1.38 + index * 0.55
        scene.add(divider)
        top = 1.85 - index * 0.106
        stock = gfx.MeshPhongMaterial(color=colour, shininess=6)
        box(divider, (4.16, top - 0.48, 0.067), (0, (top + 0.48) / 2, 0), stock)
        tab_x = -1.6 + index * 0.63
        box(divider, (0.82, 0.21, 0.067), (tab_x, top + 0.102, 0), stock)
        if index == 2:
            box(divider, (0.82, 0.058, 0.078), (tab_x, top + 0.19, 0), brass)

    scene.add(gfx.AmbientLight(intensity=0.62))
    key = gfx.DirectionalLight(color="#fff1d9", intensity=2.05)
    key.local.position = (2, 6, 4)
    scene.add(key)
    fill = gfx.DirectionalLight(color="#c3d7d4", intensity=0.58)
    fill.local.position = (-4, 3, -3)
    scene.add(fill)
    return scene, lifted


def save_frame(canvas, path):
    pixels = canvas.draw()
    if pixels is None or pixels.shape != (HEIGHT, WIDTH, 4):
        raise RuntimeError("The offscreen renderer did not return an RGBA image")
    foreground = QImage(pixels.tobytes(), WIDTH, HEIGHT, WIDTH * 4, QImage.Format.Format_RGBA8888)
    image = QImage(WIDTH, HEIGHT, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.save()
    painter.translate(322, 346)
    painter.scale(1, 0.26)
    shadow = QRadialGradient(0, 0, 242)
    shadow.setColorAt(0, QColor(0, 0, 0, 125))
    shadow.setColorAt(0.6, QColor(0, 0, 0, 60))
    shadow.setColorAt(1, QColor(0, 0, 0, 0))
    painter.setBrush(shadow)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(QRectF(-242, -242, 484, 484))
    painter.restore()
    painter.drawImage(0, 0, foreground)
    painter.end()
    if not image.save(str(path), "PNG"):
        raise OSError(f"Could not write PNG to {path}")


def main():
    adapters = wgpu.gpu.enumerate_adapters_sync()
    adapter = next(
        (
            item
            for adapter_type in ("IntegratedGPU", "CPU")
            for item in adapters
            if item.info["adapter_type"] == adapter_type and item.info["backend_type"] == "D3D12"
        ),
        None,
    )
    if adapter is None:
        raise RuntimeError("An integrated or CPU D3D12 adapter is required for the asset build")
    gfx.renderers.wgpu.select_adapter(adapter)
    canvas = OffscreenRenderCanvas(size=(WIDTH, HEIGHT))
    renderer = gfx.renderers.WgpuRenderer(canvas, pixel_scale=2)
    scene, lifted = drawer()
    camera = gfx.PerspectiveCamera(32, WIDTH / HEIGHT)
    camera.local.position = (6.5, 5.3, 8.2)
    camera.look_at((0, 0.96, 0))
    canvas.request_draw(lambda: renderer.render(scene, camera))

    motion = ASSETS / "drawer-motion"
    motion.mkdir(parents=True, exist_ok=True)
    save_frame(canvas, ASSETS / "drawer-still.png")
    for frame in range(FRAME_COUNT):
        position = frame / (FRAME_COUNT - 1)
        lifted.local.y = 0.49 * (4 * position * (1 - position)) ** 2
        save_frame(canvas, motion / f"{frame:02}.png")
    canvas.close()
    print(f"Rendered {FRAME_COUNT + 1} images on {adapter.info['device']} via D3D12")


if __name__ == "__main__":
    main()
