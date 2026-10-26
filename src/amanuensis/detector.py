"""Detección de líneas y orden de lectura (port de detector.py: cajas_desde_mapa y orden_lectura).

- DetectorMini: eg2-mini det (ONNX diminuto, CPU), mismas salidas que det2.
- DetectorEG2: det2 sobre el codificador de r2 (ONNX de la demo web; pesado, mejor con GPU).
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from .lector import sesion

NUCLEO_V = 0.5


def cajas_desde_mapa(nucleo: np.ndarray, umbral=0.5, escala_inv=4.0, escala_alto=1.0):
    import cv2

    n, lab, stats, _ = cv2.connectedComponentsWithStats((nucleo > umbral).astype(np.uint8), connectivity=8)
    out = []
    for k in range(1, n):
        x0, y0, w, h = stats[k, :4]
        x1, y1 = x0 + w, y0 + h
        if w * h < 3:
            continue
        hh = h / NUCLEO_V
        cy = (y0 + y1) / 2
        r0, r1 = max(0, y0 - 2), min(nucleo.shape[0], y1 + 2)
        ventana = lab[r0:r1, x0:x1]
        pm = nucleo[r0:r1, x0:x1] * ((ventana == 0) | (ventana == k))
        alto_cols = pm.sum(0)
        if alto_cols.max() > 0:
            hh = float(np.median(alto_cols[alto_cols > 0.5 * alto_cols.max()])) / NUCLEO_V
            filas = pm.sum(1)
            cy = r0 + 0.5 + float((filas * np.arange(len(filas))).sum() / max(filas.sum(), 1e-6))
        hh *= escala_alto
        ins = 0.15 * hh
        out.append([(x0 - ins) * escala_inv, (cy - hh / 2) * escala_inv, (x1 + ins) * escala_inv, (cy + hh / 2) * escala_inv])
    return out


def corte_xy(cajas):
    """Corte XY recursivo (filas y luego columnas): el mejor en manuscritos según «maqueta»."""
    if len(cajas) <= 1:
        return list(range(len(cajas)))

    def cortar(idx, eje):
        a0, a1 = (1, 3) if eje == 0 else (0, 2)
        iv = sorted(idx, key=lambda i: cajas[i][a0])
        grupos, actual, fin = [], [iv[0]], cajas[iv[0]][a1]
        for i in iv[1:]:
            if cajas[i][a0] > fin - 0.1 * (cajas[i][a1] - cajas[i][a0]) and (eje == 0 or cajas[i][a0] > fin + 4):
                grupos.append(actual); actual = [i]
            else:
                actual.append(i)
            fin = max(fin, cajas[i][a1])
        grupos.append(actual)
        return grupos

    def rec(idx, p=0):
        if len(idx) <= 1 or p > 30:
            return sorted(idx, key=lambda i: (cajas[i][1], cajas[i][0]))
        for eje in (0, 1):
            g = cortar(idx, eje)
            if len(g) > 1:
                return [j for x in g for j in rec(x, p + 1)]
        return sorted(idx, key=lambda i: (cajas[i][1], cajas[i][0]))

    return rec(list(range(len(cajas))))


MAX_GCN = 1500  # renglones como mucho para el orden por grafo (más, el de siempre)


def modo_orden() -> str:
    """AMANUENSIS_ORDEN: «gcn» (orden por grafo disperso, orden_gcn.py, en todas las páginas y tamaños), «prensa» (el
    grafo solo en las páginas a dos columnas o más), «ligero» (el perceptrón por pares) o «columnas» (corte XY con
    canales). Vacío = «prensa»: el grafo en prensa y revistas, el perceptrón en lo demás. Sin pesos, lo de siempre."""
    m = os.environ.get("AMANUENSIS_ORDEN", "").strip().lower()
    return m if m in ("gcn", "prensa", "ligero", "columnas") else ""


def orden_gcn(cajas, tam):
    """Orden por grafo (orden_gcn.py) o None si no hay pesos, la página no cabe o falla: quien llama sigue con el suyo."""
    if tam is None or len(cajas) < 2 or len(cajas) > MAX_GCN:
        return None
    try:
        from . import orden_gcn as G

        m = G.modelo()
        if m is None:
            return None
        o = [int(i) for i in G.ordenar(m, cajas, tam)]
    except Exception:
        return None
    return o if sorted(o) == list(range(len(cajas))) else None


def orden_lectura(cajas, modo="aprendido", tam=None):
    """«aprendido» (por defecto si hay pesos y se conoce el tamaño de la página): orden por pares (orden.py), o el
    orden por grafo si AMANUENSIS_ORDEN=gcn; «gcn»: el orden por grafo y, sin él, «columnas»; «columnas»: corte XY con
    canales de columna (impreso moderno); «xy»: corte XY simple (manuscritos)."""
    if modo == "xy":
        return corte_xy(cajas)
    if modo == "aprendido" and modo_orden() == "columnas":
        modo = "columnas"
    if modo == "gcn" or (modo == "aprendido" and modo_orden() == "gcn"):
        o = orden_gcn(cajas, tam)
        if o is not None:
            return o
        if modo == "gcn":
            modo = "columnas"
    if modo == "aprendido" and tam is not None and len(cajas) > 1 and len(cajas) <= 400:
        from . import orden

        m = orden.modelo()
        if m is not None:
            return orden.ordenar(m, cajas, tam)
    if modo == "aprendido":
        modo = "columnas"
    if len(cajas) <= 1:
        return list(range(len(cajas)))
    hmed = float(np.median([c[3] - c[1] for c in cajas]))

    def cortar(idx, eje, hueco=None):
        a0, a1 = (1, 3) if eje == 0 else (0, 2)
        iv = sorted(idx, key=lambda i: cajas[i][a0])
        grupos, actual, fin = [], [iv[0]], cajas[iv[0]][a1]
        for i in iv[1:]:
            corta = cajas[i][a0] > fin - 0.1 * (cajas[i][a1] - cajas[i][a0]) if hueco is None else cajas[i][a0] > fin + hueco
            if corta:
                grupos.append(actual); actual = [i]
            else:
                actual.append(i)
            fin = max(fin, cajas[i][a1])
        grupos.append(actual)
        return grupos

    def buscar_canal(idx):
        n = len(idx)
        if n < 6:
            return None
        x0 = min(cajas[i][0] for i in idx); x1 = max(cajas[i][2] for i in idx)
        m = 0.25 * hmed
        mejor = None
        for xc in np.arange(x0 + hmed, x1 - hmed, max(2.0, hmed / 6)):
            cruzan = [i for i in idx if cajas[i][0] < xc - m and cajas[i][2] > xc + m]
            izq = sum(1 for i in idx if cajas[i][2] <= xc + m)
            der = n - len(cruzan) - izq
            if min(izq, der) < 0.2 * n or len(cruzan) > max(1, 0.08 * n):
                continue
            clave = (len(cruzan), -min(izq, der))
            if mejor is None or clave < mejor[0]:
                mejor = (clave, float(xc), cruzan)
        return None if mejor is None else (mejor[1], mejor[2])

    def rec(idx, p=0):
        if len(idx) <= 1 or p > 40:
            return sorted(idx, key=lambda i: (cajas[i][1], cajas[i][0]))
        g = cortar(idx, 1, 0.5 * hmed)
        if len(g) > 1:
            return [j for x in g for j in rec(x, p + 1)]
        canal = buscar_canal(idx)
        if canal is not None:
            xc, cruzan = canal
            cy = lambda i: (cajas[i][1] + cajas[i][3]) / 2
            cruzan.sort(key=cy)
            resto = sorted((i for i in idx if i not in cruzan), key=cy)
            out, franja, k = [], [], 0

            def volcar(fr):
                izq = [i for i in fr if (cajas[i][0] + cajas[i][2]) / 2 < xc]
                return rec(izq, p + 1) + rec([i for i in fr if i not in izq], p + 1)

            for i in resto:
                while k < len(cruzan) and cy(cruzan[k]) < cy(i):
                    out += volcar(franja) + [cruzan[k]]
                    franja, k = [], k + 1
                franja.append(i)
            return out + volcar(franja) + cruzan[k:]
        g = cortar(idx, 0, 0.7 * hmed)
        if len(g) > 1:
            return [j for x in g for j in rec(x, p + 1)]
        filas = cortar(idx, 0)
        return [j for f in filas for j in sorted(f, key=lambda i: cajas[i][0])]

    return rec(list(range(len(cajas))))


class _Base:
    ancho = 768
    multiplo = 32
    umbral = 0.5
    escala_alto = 0.8

    def _lienzo(self, pagina: Image.Image, fondo_blanco_1: bool):
        g = pagina.convert("L")
        esc = self.ancho / g.size[0]
        h = int(g.size[1] * esc)
        H = int(np.ceil(g.size[1] * esc / self.multiplo)) * self.multiplo
        r = np.asarray(g.resize((self.ancho, h), Image.BILINEAR), np.float32) / 255.0
        x = np.ones((H, self.ancho), np.float32)
        x[:h] = r
        return (x if fondo_blanco_1 else 1.0 - x), esc

    def cajas(self, nuc, esc, tam, modo="aprendido"):
        W0, H0 = tam
        cs = [[max(0, b[0] / esc), max(0, b[1] / esc), min(W0, b[2] / esc), min(H0, b[3] / esc)]
              for b in cajas_desde_mapa(nuc, self.umbral, 4.0, self.escala_alto)]
        return [cs[i] for i in orden_lectura(cs, modo, tam)]


class DetectorMini(_Base):
    def __init__(self, carpeta: str, hilos: int = 0, proveedores=None):
        c = json.load(open(os.path.join(carpeta, "config.json")))
        self.ancho, self.umbral, self.escala_alto = c.get("ancho", 768), c.get("umbral", 0.5), c.get("escala_alto", 0.8)
        self.s = sesion(os.path.join(carpeta, c.get("fichero", "det.onnx")), hilos, proveedores)

    def detectar(self, pagina: Image.Image, modo="aprendido"):
        x, esc = self._lienzo(pagina, False)
        lg = self.s.run(None, {"x": x[None, None]})[0][0, 0]
        nuc = 1.0 / (1.0 + np.exp(-lg))
        return self.cajas(nuc, esc, pagina.size, modo)


class DetectorEG2(_Base):
    """det2 de la demo (r2d2): codificador de r2 + detector_fp16."""

    def __init__(self, carpeta: str, variante="q8_b32", hilos: int = 0, proveedores=None):
        c = json.load(open(os.path.join(carpeta, "config.json")))["detector"]
        self.ancho, self.umbral, self.escala_alto = 768, c.get("umbral", 0.5), c.get("escala_alto", 0.8)
        self.multiplo = c.get("multiplo", 48)
        self.cod = sesion(os.path.join(carpeta, f"codificador_{variante}.onnx"), hilos, proveedores)
        self.det = sesion(os.path.join(carpeta, "detector_fp16.onnx"), hilos, proveedores)

    def detectar(self, pagina: Image.Image, modo="aprendido"):
        x, esc = self._lienzo(pagina, True)
        est, extra = self.cod.run(["estados", "extra"], {"pixeles": x[None]})
        tipo = self.det.get_inputs()[0].type
        dt = np.float16 if "float16" in tipo else np.float32
        nuc = self.det.run(None, {"gris": x[None, None].astype(dt), "extra": extra.astype(dt), "estados": est.astype(dt)})[0][0]
        return self.cajas(nuc.astype(np.float32), esc, pagina.size, modo)
