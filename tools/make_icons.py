"""Builds the transparent logo and the phone app icons from app/assets/logo-source.jpg."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
A = Path(__file__).resolve().parents[1] / "app" / "assets"
src = Image.open(A / "logo-source.jpg").convert("RGB")
# 1) find the white background that touches the outside, make it transparent
near_white = src.point(lambda v: 255 if v > 232 else 0).convert("L")
bg = Image.new("L", src.size, 0)
mask = near_white.copy()
for pt in [(0, 0), (src.width - 1, 0), (0, src.height - 1), (src.width - 1, src.height - 1)]:
    ImageDraw.floodfill(mask, pt, 128, thresh=0)
outer = mask.point(lambda v: 255 if v == 128 else 0)
outer = outer.filter(ImageFilter.MaxFilter(3))                     # eat the light JPEG fringe
alpha = outer.point(lambda v: 0 if v else 255).filter(ImageFilter.GaussianBlur(0.7))
rgba = src.convert("RGBA"); rgba.putalpha(alpha)
rgba = rgba.crop(rgba.getbbox())
logo = rgba.copy(); logo.thumbnail((240, 240), Image.LANCZOS); logo.save(A / "logo.png", optimize=True)

def icon(size, fill, bgcolor=(255, 255, 255, 255)):
    c = Image.new("RGBA", (size, size), bgcolor)
    l = rgba.copy(); l.thumbnail((int(size * fill), int(size * fill)), Image.LANCZOS)
    c.alpha_composite(l, ((size - l.width) // 2, (size - l.height) // 2))
    return c.convert("RGB")
icon(512, .66).save(A / "icon-512.png", optimize=True)
icon(192, .66).save(A / "icon-192.png", optimize=True)
icon(180, .66).save(A / "apple-touch-icon.png", optimize=True)
icon(512, .50).save(A / "icon-maskable-512.png", optimize=True)   # extra padding for round/squircle masks
print(logo.size, [p.name for p in sorted(A.glob("*.png"))])
