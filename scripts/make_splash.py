"""生成启动过渡图 open_spss/splash.png(520x320)。

图上不画进度条与状态文字:运行期由 launcher 通过 pyi_splash 的
update_text IPC 按真实启动阶段刷新一行文本(含字符进度条),
text_pos/text_color/text_default 在 openspss.spec 中配置。
用法:.venv/Scripts/python.exe scripts/make_splash.py
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 520, 320
OUT = os.path.join(os.path.dirname(__file__), "..", "open_spss", "splash.png")

GRAD_TOP = (13, 20, 48)
GRAD_BOTTOM = (38, 52, 112)


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.path.join("C:/Windows/Fonts", name), size)


def _gradient() -> Image.Image:
    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        t = y / (H - 1)
        r = round(GRAD_TOP[0] + (GRAD_BOTTOM[0] - GRAD_TOP[0]) * t)
        g = round(GRAD_TOP[1] + (GRAD_BOTTOM[1] - GRAD_TOP[1]) * t)
        b = round(GRAD_TOP[2] + (GRAD_BOTTOM[2] - GRAD_TOP[2]) * t)
        for x in range(W):
            px[x, y] = (r, g, b)
    return img


def _chart_art(draw: ImageDraw.ImageDraw) -> None:
    """右上角的折线图装饰,呼应统计分析主题。"""
    pts = [(322, 88), (368, 58), (404, 96), (446, 46), (492, 66)]
    draw.line(pts, fill=(124, 147, 245), width=5, joint="curve")
    for x, y in pts:
        r = 7
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(233, 238, 255))
    # 底部淡淡的柱状装饰
    bars = [(356, 24), (386, 34), (416, 18), (446, 40), (476, 28)]
    for x, h in bars:
        draw.rectangle((x, 128 - h, x + 18, 128), fill=(72, 90, 170))


def main() -> None:
    img = _gradient()
    draw = ImageDraw.Draw(img)

    _chart_art(draw)

    # 品牌字标:Open(白) + SPSS(亮蓝紫)
    bold = _font("msyhbd.ttc", 52)
    draw.text((44, 108), "Open", font=bold, fill=(245, 247, 255))
    w_open = draw.textlength("Open", font=bold)
    draw.text((44 + w_open, 108), "SPSS", font=bold, fill=(147, 168, 255))

    tag = _font("msyh.ttc", 17)
    draw.text((50, 182), "开源统计分析平台", font=tag, fill=(143, 160, 216))

    # y≈205-250 预留给运行期动态文本(启动阶段 + 字符进度条),此处刻意留空
    draw.text((50, 276), "open-spss", font=_font("msyh.ttc", 13), fill=(96, 112, 178))

    img.save(os.path.normpath(OUT))
    print("splash.png written")


if __name__ == "__main__":
    main()
