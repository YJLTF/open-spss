"""生成应用图标 open_spss/logo.ico(与 web/favicon.svg 同一品牌设计)。

圆角方块对角渐变(#5B6BFF→#3F8BFF)+ 白色上扬折线,2048px 超采样后缩到各尺寸。
用法:.venv/Scripts/python.exe scripts/make_logo.py
"""
import math
import os

from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(__file__), "..", "open_spss", "logo.ico")
S = 2048          # 画布
M = 256           # 边距(对应 viewBox 2/24)
D = S - 2 * M     # 方块边长
RX = D // 4       # 圆角(viewBox rx=6,约占边长 1/4)
C1, C2 = (91, 107, 255), (63, 139, 255)


def rounded_gradient() -> Image.Image:
    tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((M, M, M + D, M + D), radius=RX, fill=255)
    grad = Image.new("RGBA", (S, S))
    px = grad.load()
    for y in range(S):
        for x in range(0, S, 4):
            t = (x + y) / (2 * S)
            c = tuple(round(a + (b - a) * t) for a, b in zip(C1, C2)) + (255,)
            for dx in range(4):
                if x + dx < S:
                    px[x + dx, y] = c
    tile.paste(grad, (0, 0), mask)
    return tile


def draw_polyline(img: Image.Image) -> None:
    """viewBox 路径 M7 14.5 L10.2 10.2 L12.8 13.3 L17 7.3,白色圆头折线。"""
    d = ImageDraw.Draw(img)
    unit = D / 20.0

    def pt(vx: float, vy: float) -> tuple:
        return (M + (vx - 2) * unit, M + (vy - 2) * unit)

    pts = [pt(7, 14.5), pt(10.2, 10.2), pt(12.8, 13.3), pt(17, 7.3)]
    w = round(2 * unit)
    d.line(pts, fill=(255, 255, 255, 255), width=w, joint="curve")
    r = w // 2
    for p in (pts[0], pts[-1]):
        d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(255, 255, 255, 255))


def main() -> None:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    img = rounded_gradient()
    draw_polyline(img)
    img.resize((256, 256), Image.LANCZOS).save(
        os.path.normpath(OUT), sizes=[(n, n) for n in sizes])
    print("logo.ico written")


if __name__ == "__main__":
    main()
