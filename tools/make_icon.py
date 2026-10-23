"""Generate the AI Recogniser application icon.

Produces:
    shell/assets/icon.png  - 256x256 RGBA master for the window / tray
    shell/assets/icon.ico  - multi-size Windows icon (16..256)

Design: a rounded square with a blue-to-teal diagonal gradient
(#0F6CBD -> #12B5A5) and a white 5-bar waveform. Everything outside the
rounded square is transparent.

Usage:
    .\\.venv\\Scripts\\python.exe tools\\make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "shell" / "assets"

MASTER_SIZE = 1024
GRADIENT_SIZE = 256
GRADIENT_START = (0x0F, 0x6C, 0xBD)
GRADIENT_END = (0x12, 0xB5, 0xA5)
CORNER_RADIUS = 0.24  # fraction of the icon side

BAR_WIDTH = 0.10  # fraction of the icon side
BAR_GAP = 0.05
BAR_HEIGHTS = (0.30, 0.55, 0.78, 0.55, 0.30)

ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def _diagonal_gradient(size: int) -> Image.Image:
    """Blue -> teal gradient running from the top-left to the bottom-right."""
    small = Image.new("RGB", (GRADIENT_SIZE, GRADIENT_SIZE))
    pixels = small.load()
    span = 2 * (GRADIENT_SIZE - 1)
    for y in range(GRADIENT_SIZE):
        for x in range(GRADIENT_SIZE):
            t = (x + y) / span
            pixels[x, y] = tuple(
                round(start + (end - start) * t)
                for start, end in zip(GRADIENT_START, GRADIENT_END, strict=True)
            )
    return small.resize((size, size), Image.Resampling.BICUBIC)


def _rounded_mask(size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    radius = round(size * CORNER_RADIUS)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def _waveform_mask(size: int) -> Image.Image:
    """Five centred bars with rounded caps, tallest in the middle."""
    layer = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(layer)
    bar_width = size * BAR_WIDTH
    gap = size * BAR_GAP
    total = len(BAR_HEIGHTS) * bar_width + (len(BAR_HEIGHTS) - 1) * gap
    x = (size - total) / 2
    centre_y = size / 2
    for height in BAR_HEIGHTS:
        half = size * height / 2
        draw.rounded_rectangle(
            (x, centre_y - half, x + bar_width, centre_y + half),
            radius=bar_width / 2,
            fill=255,
        )
        x += bar_width + gap
    return layer


def build_master() -> tuple[Image.Image, Image.Image]:
    """Return the opaque artwork and its rounded-square alpha mask."""
    artwork = _diagonal_gradient(MASTER_SIZE)
    artwork.paste(
        Image.new("RGB", artwork.size, (255, 255, 255)),
        (0, 0),
        _waveform_mask(MASTER_SIZE),
    )
    return artwork, _rounded_mask(MASTER_SIZE)


def render(size: int, artwork: Image.Image, mask: Image.Image) -> Image.Image:
    # Resize colour and alpha separately so transparent corners never bleed
    # black into the antialiased edge.
    icon = artwork.resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")
    icon.putalpha(mask.resize((size, size), Image.Resampling.LANCZOS))
    return icon


def main() -> int:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    artwork, mask = build_master()

    icon_256 = render(256, artwork, mask)
    png_path = ASSETS_DIR / "icon.png"
    ico_path = ASSETS_DIR / "icon.ico"

    icon_256.save(png_path, format="PNG", optimize=True)
    icon_256.save(
        ico_path,
        format="ICO",
        sizes=[(size, size) for size in ICON_SIZES],
    )

    print(f"wrote {png_path} (256x256)")
    print(f"wrote {ico_path} ({len(ICON_SIZES)} sizes: {ICON_SIZES[0]}..{ICON_SIZES[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
