"""«Ligero prensa»: el lector de Ligero (mini.onnx) leyendo los renglones que encuentra la maquetación del modelo completo
(maquetación, detector de renglones i50 y orden v5p en Core ML u ONNX). Se usa con `-m ligero --prensa si` si el modelo completo está.

Adapta LectorMini al contrato del lector de `Completo`: `leer(recortes, presupuesto, sesgo) -> [(texto, conf, confs, fotogramas)]`
con recortes [alto, w16] en 0-255 ya enderezados y normalizados; los fotogramas son de 4 px de recorte (columnas de 16 px y 4 por columna).
Ligero espera recortes con margen (0,25 alturas arriba y abajo, 0,6 a los lados) que él mismo recorta: se les añade ese margen de papel.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


class LectorLigeroEnCompleto:
    def __init__(self, lector, alto: int = 96, max_cols: int = 240):
        self.l, self.alto, self.max_cols = lector, alto, max_cols
        self.variante = "ligero"
        self.haz = self.corrector = None  # el haz de Ligero actúa dentro de lector.voraz (HAZ_CLASICO)

    def leer(self, recortes, presupuesto=0, sesgo=0.0):
        pils, pad = [], []
        for a in recortes:
            a = np.asarray(a, np.uint8)
            h, w = a.shape
            py, px = int(round(0.25 * h)), int(round(0.6 * h))
            g = np.full((h + 2 * py, w + 2 * px), 255, np.uint8)
            g[py:py + h, px:px + w] = a
            pils.append(Image.fromarray(g))
            pad.append(px)
        out = []
        for ll, px in zip(self.l.leer(pils, sesgo=sesgo), pad):
            out.append((ll.texto, ll.conf, list(ll.confs), [(x - px) / 4.0 - 0.5 for x in ll.xs]))
        return out
