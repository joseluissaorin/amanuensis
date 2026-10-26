"""Rectificación por tiras: el renglón de una página curvada o torcida, enderezado antes de leerlo.

En una fotocopia de un libro abierto los renglones se curvan hacia el lomo y suben o bajan a lo ancho de la página: la
caja recta del detector corta en diagonal y mezcla trozos de dos renglones («caummanicación» por «la comunicación»). Aquí
se sigue la tinta del renglón tira a tira (con continuidad: cada tira busca cerca de la anterior), se ajusta una curva
suave a su centro y se rehace el recorte columna a columna con ese centro en medio. El recorte empieza en el mismo x que
`imagen.recorte_de_caja`, así que las posiciones de las letras valen igual.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .imagen import MARGEN_H, MARGEN_V

DESVIO_MIN = 0.15  # en cuerpos: por debajo, el renglón ya va derecho y no se relee

def rectificar(hoja: Image.Image, caja, alto_banda: float = 1.6, tira: float = 1.5, margen_h: float = MARGEN_H):
    """(recorte enderezado, desvío del renglón en cuerpos); (None, 0) si no hay tinta que seguir."""
    x0, y0, x1, y1 = [float(v) for v in caja[:4]]
    h = max(4.0, y1 - y0)
    X0, X1 = int(max(0, x0 - margen_h * h)), int(min(hoja.width, x1 + margen_h * h))
    Y0, Y1 = int(max(0, y0 - alto_banda * h)), int(min(hoja.height, y1 + alto_banda * h))
    g = np.asarray(hoja.convert("L").crop((X0, Y0, X1, Y1)), np.float32)
    tinta = np.clip((200 - g) / 200, 0, 1)
    H, W = tinta.shape
    cy0 = (y0 + y1) / 2 - Y0
    paso = max(8, int(tira * h))
    centros, xs = [], []
    # empezar por la tira con más tinta cerca del centro de la caja y seguir a izquierda y derecha
    nt = max(1, W // paso)
    perf = [tinta[:, k * paso:(k + 1) * paso].sum(1) for k in range(nt)]
    def pico(p, cerca, rad):
        lo, hi = int(max(0, cerca - rad)), int(min(H, cerca + rad + 1))
        if hi <= lo or p[lo:hi].sum() < 1e-3:
            return None
        ys = np.arange(lo, hi)
        w = p[lo:hi] ** 2
        return float((ys * w).sum() / w.sum())
    k0 = int(np.argmax([p[int(max(0, cy0 - h / 2)):int(cy0 + h / 2)].sum() for p in perf]))
    c = {k0: pico(perf[k0], cy0, 0.5 * h) or cy0}
    for rango in (range(k0 + 1, nt), range(k0 - 1, -1, -1)):
        ant = c[k0]
        for k in rango:
            v = pico(perf[k], ant, 0.35 * h)
            if v is not None:
                c[k] = v; ant = v
    ks = sorted(c)
    if len(ks) < 3:
        return None, 0.0
    xk = np.array([(k + 0.5) * paso for k in ks]); yk = np.array([c[k] for k in ks])
    coef = np.polyfit(xk, yk, 2 if len(ks) >= 5 else 1)
    curva = np.polyval(coef, np.arange(W))
    desvio = float(np.ptp(curva))
    alto_out = int(h * (1 + 2 * MARGEN_V)) + 2
    ys = np.arange(alto_out)[:, None] + (curva - alto_out / 2)[None, :]  # fila de origen de cada píxel de salida
    ok = (ys >= 0) & (ys < H - 1)
    yi = np.clip(ys.astype(int), 0, H - 2)
    f = ys - yi
    cols = np.broadcast_to(np.arange(W)[None, :], ys.shape)
    out = g[yi, cols] * (1 - f) + g[yi + 1, cols] * f
    out = np.where(ok, out, 255.0)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)), desvio / h
