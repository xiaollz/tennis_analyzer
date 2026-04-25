#!/usr/bin/env python3
"""Generate PWA icons for Baseline.

Produces:
  frontend/dist/icon-192.png        (Android, 192x192)
  frontend/dist/icon-512.png        (Android, 512x512)
  frontend/dist/icon-maskable.png   (Android maskable, 512x512 with safe zone)
  frontend/dist/apple-touch-icon.png (iOS, 180x180)
  frontend/dist/favicon.png         (browser tab, 32x32)

Design follows the Baseline wordmark in tennis/project/screens/Onboarding.jsx:
  • Square with `clay` background
  • Centered `amber` circle
  • Slight inner padding for "maskable" version
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw

CLAY = (200, 85, 61, 255)      # #C8553D
AMBER = (232, 176, 75, 255)    # #E8B04B
PAPER = (247, 243, 236, 255)   # #F7F3EC
INK = (42, 41, 37, 255)         # #2A2925

OUT = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def base_icon(size: int, padding: float = 0.0, bg=CLAY, fg=AMBER) -> Image.Image:
    """Square icon with a centered amber circle on a clay background.

    padding: 0.0–0.3 — fraction of size to leave as inner safe zone.
    """
    img = Image.new("RGBA", (size, size), bg)
    draw = ImageDraw.Draw(img)

    # Optionally inset the artwork (for maskable icons)
    inset = int(size * padding)
    inner = size - 2 * inset

    # Amber dot — diameter ~ 38% of inner area
    dot_d = int(inner * 0.38)
    cx = size // 2
    cy = size // 2
    draw.ellipse(
        (cx - dot_d // 2, cy - dot_d // 2, cx + dot_d // 2, cy + dot_d // 2),
        fill=fg,
    )

    # Subtle paper-ring around the dot (highlights the wordmark feel)
    ring_d = int(inner * 0.50)
    ring_thickness = max(2, size // 96)
    draw.ellipse(
        (cx - ring_d // 2, cy - ring_d // 2, cx + ring_d // 2, cy + ring_d // 2),
        outline=(247, 243, 236, 64),
        width=ring_thickness,
    )

    return img


def rounded_corners(img: Image.Image, radius: int) -> Image.Image:
    """Apply rounded corners (for iOS apple-touch-icon-style result).

    iOS auto-rounds touch icons but rounding pre-emptively avoids weird
    aliasing when the user adds to home screen on Android Chrome.
    """
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, img.size[0], img.size[1]),
                           radius=radius, fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 192 — Android any
    img192 = base_icon(192).convert("RGBA")
    img192 = rounded_corners(img192, radius=int(192 * 0.18))
    img192.save(OUT / "icon-192.png", optimize=True)

    # 512 — Android any (also serves as splash)
    img512 = base_icon(512).convert("RGBA")
    img512 = rounded_corners(img512, radius=int(512 * 0.18))
    img512.save(OUT / "icon-512.png", optimize=True)

    # Maskable — 512 with full bleed (no rounding) but artwork inset 10%
    # so launcher masks don't crop the dot
    img_mask = base_icon(512, padding=0.10).convert("RGBA")
    img_mask.save(OUT / "icon-maskable.png", optimize=True)

    # Apple touch icon — 180 (iOS standard, no rounding needed)
    img180 = base_icon(180).convert("RGBA")
    img180.save(OUT / "apple-touch-icon.png", optimize=True)

    # Favicon — 32 (browser tab)
    img32 = base_icon(32).convert("RGBA")
    img32.save(OUT / "favicon.png", optimize=True)

    for f in ("icon-192.png", "icon-512.png", "icon-maskable.png",
              "apple-touch-icon.png", "favicon.png"):
        p = OUT / f
        print(f"✓ {p.name}  ({p.stat().st_size} B)")


if __name__ == "__main__":
    main()
