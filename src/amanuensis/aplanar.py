"""Aplanado de fotos de documentos con UVDoc (MIT, github.com/tanguymagne/UVDoc; licencia en LICENSE-UVDoc.txt junto al modelo)
exportado a ONNX (8 M de parámetros), con selector «aplanar solo si hace falta». De maqueta/blender/uvdoc_aplanar.py (blender).
Necesita onnxruntime y opencv (opcional: sin opencv `--aplanar` avisa y la lectura sigue sin aplanar).

Selector: D = residuo RMS de la rejilla 45 × 31 que predice UVDoc tras ajustarle la mejor homografía, relativo a la diagonal. Si
D <= UMBRAL la página ya es (casi) plana y se devuelve tal cual. UMBRAL = 0,0058, elegido en 120 fotos simuladas, no en el banco real.
Las cajas de la salida van sobre la imagen aplanada, no sobre la foto original.
"""
from __future__ import annotations

import time

import numpy as np

ANCHO, ALTO = 488, 712   # entrada de UVDoc (IMG_SIZE)
UMBRAL = 0.0058


def es_foto(rgb: np.ndarray) -> bool:
    """¿Parece una foto (y no un escaneo)? Rasgos baratos sobre una miniatura de 320 px: color del marco (los escaneos de archivo
    son grises, el fondo de una foto no) y marco distinto del centro o iluminación desigual. Medido en 146 fotos (16 del banco +
    130 de DocUNet) y 165 escaneos (90 de val de GRESEL, 10 de La Vanguardia, 65 de DocUNet): reconoce 143 de 146 fotos y confunde
    18 de 165 escaneos; junto con el selector de deformación (D) solo deja pasar 2 de 165 escaneos y todas las fotos que el
    selector aplana."""
    import cv2

    h0, w0 = rgb.shape[:2]
    e = 320 / max(h0, w0)
    a = cv2.resize(rgb, (max(1, round(w0 * e)), max(1, round(h0 * e))), interpolation=cv2.INTER_AREA) if e < 1 else rgb
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    h, w = g.shape
    b = max(2, int(0.05 * min(h, w)))
    anillo = np.ones((h, w), bool)
    anillo[b:-b, b:-b] = False
    sat = float(hsv[..., 1][anillo].mean())
    dif = abs(float(g[anillo].mean()) - float(g[~anillo].mean()))
    mx = np.array([np.percentile(g[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4], 90) for i in range(4) for j in range(4)])
    ilum = float(mx.std() / max(mx.mean(), 1))
    tinte = float(np.std(np.percentile(a.reshape(-1, 3), 90, axis=0)))
    return bool((sat > 15 or tinte > 3) and (dif > 30 or ilum > 0.03))


def residuo_homografia(g):
    import cv2

    """g (gh, gw, 2): RMS del residuo tras la mejor homografía desde la cuadrícula regular, relativo a la diagonal."""
    gh, gw = g.shape[:2]
    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float64)
    src = np.stack([gx, gy], -1).reshape(-1, 2)
    dst = g.reshape(-1, 2).astype(np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    p = cv2.perspectiveTransform(src[None], H)[0]
    diag = np.linalg.norm(dst.max(0) - dst.min(0))
    return float(np.sqrt(((p - dst) ** 2).sum(1).mean()) / max(diag, 1e-9))


class AplanadorUVDoc:
    def __init__(self, ruta, hilos=4, coreml=True):
        import cv2  # noqa: F401  (falla aquí si falta opencv)
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = hilos
        prov = (["CoreMLExecutionProvider"] if coreml and "CoreMLExecutionProvider" in ort.get_available_providers() else []) + ["CPUExecutionProvider"]
        self.s = ort.InferenceSession(ruta, so, providers=prov)

    def rejilla(self, rgb):
        import cv2
        # bilineal, como la demo de UVDoc y como se calibró el umbral
        x = cv2.resize(rgb.astype(np.float32) / 255.0, (ANCHO, ALTO)).transpose(2, 0, 1)[None]
        g2, _ = self.s.run(None, {"foto": x})
        return g2[0].transpose(1, 2, 0)  # (45, 31, 2): x, y en -1..1 de la foto

    def __call__(self, rgb, modo="auto", umbral=UMBRAL):
        t0 = time.time()
        if modo == "auto" and not es_foto(rgb):  # un escaneo no se toca
            return rgb, {"deformacion": None, "umbral": umbral, "aplanada": False, "foto": False, "ms": round((time.time() - t0) * 1000)}
        g = self.rejilla(rgb)
        D = residuo_homografia(g)
        info = {"deformacion": round(D, 5), "umbral": umbral, "foto": True}
        if modo == "no" or (modo == "auto" and D <= umbral):
            info.update(aplanada=False, ms=round((time.time() - t0) * 1000))
            return rgb, info
        import cv2

        H, W = rgb.shape[:2]
        gh, gw = g.shape[:2]
        # como bilinear_unwarping de UVDoc (align_corners=True): rejilla a tamaño completo y muestreo
        X, Y = np.meshgrid(np.linspace(0, gw - 1, W, dtype=np.float32), np.linspace(0, gh - 1, H, dtype=np.float32))
        gx = cv2.remap(np.ascontiguousarray(g[..., 0], np.float32), X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        gy = cv2.remap(np.ascontiguousarray(g[..., 1], np.float32), X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mx, my = (gx + 1) / 2 * (W - 1), (gy + 1) / 2 * (H - 1)
        out = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        info.update(aplanada=True, ms=round((time.time() - t0) * 1000))
        return out, info
