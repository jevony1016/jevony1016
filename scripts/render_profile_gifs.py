"""Render the GitHub profile avatar and README banner from the 知闻药观 badge.

Usage:
    python scripts/render_profile_gifs.py [path/to/badge_lines.png]

Needs Pillow and NumPy, plus NotoSerifSC-VF.ttf and msyh.ttc from C:/Windows/Fonts.

Outputs (repo root): avatar.gif, avatar_static.png, banner.gif

Approach adapted from the 知闻药观 brand GIF script (badge v2):
the badge is split into a static layer (book, seal-script 知闻) and the
directional decoration groups around it (capsules, leaves, tablets, plus
signs, dots). One soft amber light travels clockwise around the outer ring
once per loop; the ring turns amber where the light is and each decoration
takes on the accent colour as the light passes, then returns to ink.
Nothing else moves. 6 s loop, 8 fps, frame 48 == frame 0, so it is seamless.
"""
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
# Unmodified copy of the 知闻药观 avatar line art (#062D23 lines on transparent, 640x640).
DEFAULT_LOGO = Path(__file__).resolve().parent / "assets" / "badge_lines.png"
FONTS = "C:/Windows/Fonts/"

INK = np.array((6, 45, 35), float)        # chalkboard green #062D23
ACCENT = np.array((200, 144, 46), float)  # 观 column amber #C8902E
WHITE = np.array((255, 255, 255), float)
MINT = (238, 245, 241)                    # 浅薄荷 #EEF5F1

FPS = 8
TOTAL = 48          # 6 s loop
SIGMA = 0.75        # angular width of the light (radians)
RING_MAX = 0.85     # how far the ring moves toward amber at the light's centre
DECO_MAX = 1.0      # decorations reach full amber


def _label(mask):
    """8-connected component labelling (the badge is only 640x640)."""
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    comps = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        n = len(comps) + 1
        q, pts = deque([(y0, x0)]), []
        lab[y0, x0] = n
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = n
                        q.append((yy, xx))
        ys, xs = zip(*pts)
        comps.append((np.array(ys), np.array(xs)))
    return comps


class Badge:
    """Splits the badge once; frame(t) returns an RGBA image at source size."""

    def __init__(self, path):
        src = Image.open(path).convert("RGBA")
        self.size = src.size
        alpha = np.asarray(src.getchannel("A")).astype(float) / 255
        self.alpha = alpha
        H, W = alpha.shape
        yy, xx = np.mgrid[0:H, 0:W]

        # Ring centre: fit from the outermost line pixels.
        c0 = W / 2
        r0 = np.hypot(xx - c0, yy - c0)
        ring_px = (alpha > 0.25) & (r0 > 255)
        self.cx = (xx[ring_px].min() + xx[ring_px].max()) / 2
        self.cy = (yy[ring_px].min() + yy[ring_px].max()) / 2
        r = np.hypot(xx - self.cx, yy - self.cy)
        self.ring_r = r[ring_px].max()
        # Pixel angle, 0 at 12 o'clock, increasing clockwise (image y points down).
        self.pix_ang = np.arctan2(xx - self.cx, -(yy - self.cy))

        comps = _label(alpha > 60 / 255)
        # Ring = the components whose pixels sit on the outer circles (both the thin inner and
        # the thick outer line), plus their 1 px anti-aliased fringe.
        ring = np.zeros(alpha.shape, bool)
        for ys, xs in comps:
            if np.median(np.hypot(xs - self.cx, ys - self.cy)) > 245:
                ring[ys, xs] = True
        ring = np.asarray(Image.fromarray(ring).filter(ImageFilter.MaxFilter(3)))
        self.ring = ring & (r > 244) & (alpha > 0)
        # Decoration groups: big pieces plus nearby specks (e.g. dots on a capsule).
        deco = [(ys, xs) for ys, xs in comps
                if 150 < math.hypot(xs.mean() - self.cx, ys.mean() - self.cy) < 245]
        big = [k for k, (ys, xs) in enumerate(deco) if len(ys) >= 80]
        groups = {k: [k] for k in big}
        for k, (ys, xs) in enumerate(deco):
            if k in groups:
                continue
            gx, gy = xs.mean(), ys.mean()
            near = min(big, key=lambda b: math.hypot(deco[b][1].mean() - gx, deco[b][0].mean() - gy))
            if math.hypot(deco[near][1].mean() - gx, deco[near][0].mean() - gy) < 60:
                groups[near].append(k)
            else:
                groups[k] = [k]
        self.groups = []  # (angle, boolean mask incl. 1px anti-aliased fringe)
        for members in groups.values():
            m = np.zeros(alpha.shape, bool)
            for k in members:
                m[deco[k][0], deco[k][1]] = True
            m = np.asarray(Image.fromarray(m).filter(ImageFilter.MaxFilter(3))) & ~self.ring
            ys, xs = np.nonzero(m)
            ang = math.atan2(xs.mean() - self.cx, -(ys.mean() - self.cy))
            self.groups.append((ang, m))

    @staticmethod
    def lit(diff):
        d = (diff + math.pi) % (2 * math.pi) - math.pi
        return np.exp(-(d / SIGMA) ** 2)

    def frame(self, t, disc_margin=None):
        """RGBA badge at source size. Lines over white; outside the disc transparent if
        disc_margin is given (white disc of radius ring_r + margin), else white everywhere."""
        sweep = 2 * math.pi * t / TOTAL  # starts at 12 o'clock, clockwise
        mixf = np.zeros(self.alpha.shape)
        mixf[self.ring] = RING_MAX * self.lit(self.pix_ang[self.ring] - sweep)
        for ang, m in self.groups:
            mixf[m] = DECO_MAX * float(self.lit(np.array(ang - sweep)))
        col = INK + (ACCENT - INK) * mixf[..., None]
        a = self.alpha[..., None]
        rgb = WHITE * (1 - a) + col * a
        out = np.dstack([rgb, np.full(self.alpha.shape, 255.0)])
        if disc_margin is not None:
            out[..., 3] = self._disc(disc_margin)
        return Image.fromarray(np.clip(out + 0.5, 0, 255).astype("uint8"), "RGBA")

    def _disc(self, margin, ss=4):
        W, H = self.size
        big = Image.new("L", (W * ss, H * ss), 0)
        R = self.ring_r + margin
        ImageDraw.Draw(big).ellipse(
            ((self.cx + 0.5 - R) * ss, (self.cy + 0.5 - R) * ss,
             (self.cx + 0.5 + R) * ss, (self.cy + 0.5 + R) * ss), fill=255)
        return np.asarray(big.resize((W, H), Image.LANCZOS)).astype(float)


def resize_around(img, cx, cy, half_src, out):
    """Square crop of half-size half_src around (cx, cy), resized to out x out. Pads first
    (white for RGB, transparent for RGBA) when the crop reaches past the source edges."""
    pad = math.ceil(max(0, half_src - min(cx, cy, img.width - cx, img.height - cy))) + 2
    fill = (255, 255, 255) if img.mode == "RGB" else (0, 0, 0, 0)
    big = Image.new(img.mode, (img.width + 2 * pad, img.height + 2 * pad), fill)
    big.paste(img, (pad, pad))
    img, cx, cy = big, cx + pad, cy + pad
    box = (cx + 0.5 - half_src, cy + 0.5 - half_src, cx + 0.5 + half_src, cy + 0.5 + half_src)
    return img.resize((out, out), Image.LANCZOS, box=box)


def save_gif(frames, path, colors=160):
    # Palette from a mosaic of every 4th frame so the accent (present only in some frames)
    # makes it in; no dithering, so flat areas stay clean.
    picks = frames[::4]
    strip = Image.new("RGB", (picks[0].width, sum(f.height for f in picks)))
    y = 0
    for f in picks:
        strip.paste(f, (0, y))
        y += f.height
    pal = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=[120, 130] * (len(q) // 2),  # GIF timing is in 10 ms steps; 120/130 averages 125 ms = 8 fps, 6.0 s loop
             
              loop=0, optimize=True, disposal=1)
    print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB, {len(frames)} frames")


def avatar(b, size=460, ring_d=392):
    """Square avatar, white background. Ring diameter ring_d leaves (size-ring_d)/2 px to the
    inscribed circle GitHub crops to."""
    half_src = (size / 2) * (2 * b.ring_r) / ring_d
    frames = []
    for t in range(TOTAL):
        f = b.frame(t).convert("RGB")
        frames.append(resize_around(f, b.cx, b.cy, half_src, size))
    return frames


def banner(b, W=1200, H=300, S=2):
    serif = ImageFont.truetype(FONTS + "NotoSerifSC-VF.ttf", 70 * S)
    serif.set_variation_by_name("SemiBold")
    sub_font = ImageFont.truetype(FONTS + "msyh.ttc", 30 * S)
    title, sub = "jevony1016", "Clinical research × AI · 知闻药观"
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    tw = max(probe.textlength(title, font=serif), probe.textlength(sub, font=sub_font)) / S

    ring_d, gap, margin = 212, 52, 10   # badge ring diameter, badge-text gap, white disc margin (src px)
    disc_d = ring_d * (1 + 2 * margin / (2 * b.ring_r))
    group_w = disc_d + gap + tw
    x0 = (W - group_w) / 2
    bcx, bcy = x0 + disc_d / 2, H / 2
    tx = x0 + disc_d + gap

    base = Image.new("RGBA", (W * S, H * S), MINT + (255,))
    d = ImageDraw.Draw(base)
    ink = tuple(int(v) for v in INK)
    sub_col = (46, 92, 78)
    d.text((tx * S, (H / 2 - 14) * S), title, font=serif, fill=ink + (255,), anchor="ls")
    d.text((tx * S, (H / 2 + 52) * S), sub, font=sub_font, fill=sub_col + (255,), anchor="ls")

    # Badge cell: square around ring centre, big enough for the disc plus 2 px.
    cell = math.ceil(disc_d / 2 + 2) * 2
    half_src = (cell / 2) * (2 * b.ring_r) / ring_d
    frames = []
    for t in range(TOTAL):
        f = b.frame(t, disc_margin=margin)
        badge_img = resize_around(f, b.cx, b.cy, half_src, cell * S)
        img = base.copy()
        img.alpha_composite(badge_img, (round((bcx - cell / 2) * S), round((bcy - cell / 2) * S)))
        frames.append(img.convert("RGB").resize((W, H), Image.LANCZOS))
    return frames


def main():
    logo = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOGO
    b = Badge(logo)
    print(f"ring centre ({b.cx:.1f}, {b.cy:.1f}), radius {b.ring_r:.1f}, {len(b.groups)} decoration groups")
    av = avatar(b)
    save_gif(av, ROOT / "avatar.gif")
    av[0].save(ROOT / "avatar_static.png", optimize=True)
    print("avatar_static.png:", f"{(ROOT / 'avatar_static.png').stat().st_size / 1024:.0f} KB")
    save_gif(banner(b), ROOT / "banner.gif")


if __name__ == "__main__":
    main()
