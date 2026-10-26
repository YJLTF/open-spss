"""生成启动过渡图 open_spss/splash.png(520x320,2x 超采样抗锯齿)。

模型数量与版本号不画死:分别取自 open_spss 模型注册表与包 __version__,
openspss.spec 在每次构建前自动重新生成本图,新增模型/升版本无需改图。
y≈208-252 预留给运行期动态文本(位置与颜色在 spec 配置);y≈276 为页脚。
用法:.venv/Scripts/python.exe scripts/make_splash.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import make_logo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

W, H = 520, 320
K = 2                       # 超采样倍率
W2, H2 = W * K, H * K
OUT = os.path.join(ROOT, "open_spss", "splash.png")

GRAD_TOP = (11, 18, 48)     # #0B1230
GRAD_BOTTOM = (42, 58, 126) # #2A3A7E
GLOW = (99, 124, 255)
ACCENT = (140, 160, 255)
INK_SOFT = (143, 160, 216)
INK_FAINT = (91, 111, 178)


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.path.join("C:/Windows/Fonts", name), size * K)


def _background() -> Image.Image:
    """垂直渐变 + 左上品牌辉光 + 细点阵纹理(numpy 逐像素,一次成型)。"""
    yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float64)
    t = yy / (H2 - 1)
    img = np.zeros((H2, W2, 3))
    for c in range(3):
        img[..., c] = GRAD_TOP[c] * (1 - t) + GRAD_BOTTOM[c] * t
    # 品牌辉光:以字标区为中心的柔和径向光
    dist = np.sqrt((xx - 0.28 * W2) ** 2 + (yy - 0.36 * H2) ** 2)
    glow = np.clip(1 - dist / (0.62 * W2), 0, 1) ** 2
    for c in range(3):
        img[..., c] += GLOW[c] * glow * 0.16
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def _dot_grid(base: Image.Image) -> None:
    d = ImageDraw.Draw(base, "RGBA")
    step = 26 * K
    for gy in range(step // 2, H2, step):
        for gx in range(step // 2, W2, step):
            d.ellipse((gx - 1.1 * K, gy - 1.1 * K, gx + 1.1 * K, gy + 1.1 * K),
                      fill=(165, 182, 255, 16))


def _chart_art(base: Image.Image) -> None:
    """右上角折线 + 垂直渐隐面积,呼应统计分析主题(整体位于字标上方)。"""
    pts = [(322, 66), (368, 40), (404, 74), (446, 28), (492, 48)]
    pts = [(x * K, y * K) for x, y in pts]
    fill_bottom = 96 * K
    # 面积:多边形蒙版 × 垂直渐隐的 alpha 渐变
    mask = Image.new("L", (W2, H2), 0)
    ImageDraw.Draw(mask).polygon(pts + [(pts[-1][0], fill_bottom), (pts[0][0], fill_bottom)], fill=255)
    m = np.asarray(mask, dtype=np.float64) / 255.0
    top_y = min(y for _, y in pts)
    fade = np.clip(1 - (np.arange(H2)[:, None] - top_y) / (fill_bottom - top_y), 0, 1)
    alpha = (m * fade * 92).astype(np.uint8)
    layer = Image.new("RGBA", (W2, H2), (*ACCENT, 0))
    layer.putalpha(Image.fromarray(alpha, "L"))
    d = ImageDraw.Draw(layer)
    d.line(pts, fill=(*ACCENT, 255), width=int(2.6 * K), joint="curve")
    for x, y in pts:
        r = 4.4 * K
        d.ellipse((x - r * 1.9, y - r * 1.9, x + r * 1.9, y + r * 1.9), fill=(*ACCENT, 70))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(236, 241, 255, 255))
    # 底部淡柱装饰(右移避开字标)
    for x, h in [(400, 20), (424, 28), (448, 15), (472, 32), (496, 22)]:
        d.rectangle((x * K, 116 * K - h * K, (x + 18) * K, 116 * K), fill=(72, 90, 170, 110))
    base.paste(Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB"), (0, 0))


def _models_count() -> int:
    from open_spss.models import CATEGORIES
    return sum(len(cat["models"]) for cat in CATEGORIES)


def _brand(base: Image.Image, models: int, version: str) -> None:
    d = ImageDraw.Draw(base, "RGBA")
    # 品牌图标块(与 favicon/logo.ico 同一设计)
    tile = make_logo.rounded_gradient()
    draw_poly = make_logo.draw_polyline
    draw_poly(tile)
    tile = tile.resize((96 * K, 96 * K), Image.LANCZOS)
    base.paste(tile, (44 * K, 76 * K), tile)
    # 字标
    bold = _font("msyhbd.ttc", 44)
    x0 = (44 + 96 + 20) * K
    y0 = (76 + 96 // 2 - 30) * K
    d.text((x0, y0), "Open", font=bold, fill=(245, 247, 255, 255))
    w_open = d.textlength("Open", font=bold)
    d.text((x0 + w_open, y0), "SPSS", font=bold, fill=(147, 168, 255, 255))
    # 标语 + 分隔细线
    d.rectangle((46 * K, 182 * K, 78 * K, 184 * K), fill=(*ACCENT, 200))
    tag = _font("msyh.ttc", 16)
    d.text((88 * K, 174 * K), f"开源统计分析平台 · {models} 个分析模型",
           font=tag, fill=(*INK_SOFT, 255))
    # 页脚(中带 208-252 留空给动态文本)
    foot = _font("msyh.ttc", 13)
    d.text((50 * K, 270 * K), "open-spss", font=foot, fill=(*INK_FAINT, 255))
    ver = f"v{version}"
    wv = d.textlength(ver, font=foot)
    d.text(((W - 50) * K - wv, 270 * K), ver, font=foot, fill=(*INK_FAINT, 255))


def main(models: int | None = None, version: str | None = None) -> None:
    models = _models_count() if models is None else models
    if version is None:
        from open_spss import __version__
        version = __version__
    img = _background()
    _dot_grid(img)
    _chart_art(img)
    _brand(img, models, version)
    img.resize((W, H), Image.LANCZOS).save(os.path.normpath(OUT))
    print(f"splash.png written (models={models}, version=v{version})")


if __name__ == "__main__":
    main()
