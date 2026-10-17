"""Preparación de páginas y líneas, idéntica a la del entrenamiento (leer.py, entreno2/prep.py,
entreno2/aumentos2.py) y a la de la demo web (web/src/lector.ts)."""
from __future__ import annotations

import numpy as np
from PIL import Image

MARGEN_V, MARGEN_H = 0.25, 0.6  # recorte de línea desde su caja (corpus.py)


def gris(img) -> Image.Image:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.convert("L")


def recorte_de_caja(pagina: Image.Image, caja) -> Image.Image:
    x0, y0, x1, y1 = caja[:4]
    h = max(1.0, y1 - y0)
    return pagina.crop((int(x0 - MARGEN_H * h), int(y0 - MARGEN_V * h), int(x1 + MARGEN_H * h), int(y1 + MARGEN_V * h)))


def normalizar_linea(img: Image.Image) -> Image.Image:
    a = np.asarray(img.convert("L"), dtype=np.float32)
    if a.size == 0:
        return img.convert("L")
    borde = np.concatenate([a[:2].ravel(), a[-2:].ravel(), a[:, :2].ravel(), a[:, -2:].ravel()])
    fondo = np.median(borde)
    if fondo < np.median(a) - 5 or fondo < 100:
        a = 255.0 - a
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    if hi - lo > 8:
        a = (a - lo) * (255.0 / (hi - lo))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def recortar_fijo(img: Image.Image, v: float, hz: float, trunc: bool = False) -> Image.Image:
    if not (v or hz):
        return img
    w, h = img.size
    if trunc:
        dy, dx = int(h * v), int(h * hz)
        if h - 2 * dy >= 8 and w - 2 * dx >= 8:
            return img.crop((dx, dy, w - dx, h - dy))
        return img
    dy, dx = int(round(h * v)), int(round(h * hz))
    if w - 2 * dx < max(8, w // 2):
        dx = 0
    return img.crop((dx, dy, w - dx, h - dy))


def _fondo(img: Image.Image) -> int:
    return 255 if np.asarray(img).mean() > 128 else 0


def preparar(img: Image.Image, alto: int, max_cols: int = 160) -> Image.Image:
    """entrenar.preparar sin aumentos (lector r1/r2): recorta a max_cols columnas de 16 px."""
    w, h = img.size
    nw = max(16, round(w * alto / max(1, h)))
    img = img.resize((nw, alto), Image.LANCZOS)
    cols = min(max_cols, -(-nw // 16))
    lienzo = Image.new("L", (cols * 16, alto), _fondo(img))
    lienzo.paste(img.crop((0, 0, min(nw, cols * 16), alto)), (0, 0))
    return lienzo


def preparar2(img: Image.Image, alto: int, max_cols: int = 240) -> Image.Image:
    """entreno2/prep.preparar2 sin aumentos (v2b96, ml_v1): estrecha en vez de recortar."""
    w, h = img.size
    nw = max(16, round(w * alto / max(1, h)))
    nw = min(nw, max_cols * 16)
    img = img.resize((nw, alto), Image.LANCZOS)
    cols = -(-nw // 16)
    lienzo = Image.new("L", (cols * 16, alto), _fondo(img))
    lienzo.paste(img, (0, 0))
    return lienzo


def fondo_derecho(a: np.ndarray) -> int:
    return 255 if a[:, -4:].mean() > 128 else 0
