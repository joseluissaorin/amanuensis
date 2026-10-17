"""Lectores con ONNX Runtime.

- LectorEG2: codificador de visión de EmbeddingGemma 2 recortado + cabeza CTC (los ONNX de la demo
  web: r2d2, v2b96, ml_v1). En CPU usa la variante de 8 bits; con CUDA o Core ML, la que diga el catálogo.
- LectorMini: eg2-mini (ONNX de un solo fichero, CPU).
Los dos devuelven, por línea, (texto, confianza media, posiciones x de cada carácter en el recorte).
"""
from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import imagen as I
from .modelos import entorno


def sesion(ruta: str, hilos: int = 0, proveedores: list | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if hilos:
        so.intra_op_num_threads = hilos
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False  # con lotes de forma variable la arena crecía sin parar
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(ruta, so, providers=proveedores or ["CPUExecutionProvider"])


@dataclass
class LineaLeida:
    texto: str
    conf: float
    xs: list  # posición x de cada carácter, en px del recorte original


def voraz(lp: np.ndarray, chars: list[str], logits: bool = False) -> LineaLeida:
    """lp [T, C] (log-probabilidades, o logits si logits=True) -> texto, confianza media de los picos
    y x por carácter (en fotogramas)."""
    ids = lp.argmax(-1)
    if logits:
        z = lp - lp.max(-1, keepdims=True)
        p = 1.0 / np.exp(z).sum(-1)
    else:
        p = np.exp(lp.max(-1))
    out, xs, confs, prev = [], [], [], 0
    for t, i in enumerate(ids):
        if i != prev and i != 0:
            out.append(chars[i - 1]); xs.append(t); confs.append(float(p[t]))
        prev = i
    return LineaLeida("".join(out), float(np.mean(confs)) if confs else 0.0, xs)


class LectorEG2:
    def __init__(self, carpeta: str, cabeza: str = "literal", variante: str = "q8_b32", hilos: int = 0,
                 proveedores: list | None = None):
        self.cfg = json.load(open(os.path.join(carpeta, "config.json")))
        c = self.cfg
        self.chars = c["chars"] if isinstance(c["chars"], list) else json.loads(c["chars"])
        nombre_cab = (c.get("cabezas") or {}).get(cabeza, "cabeza")
        self.cod = sesion(os.path.join(carpeta, f"codificador_{variante}.onnx"), hilos, proveedores)
        self.cab = sesion(os.path.join(carpeta, f"{nombre_cab}_{variante}.onnx"), hilos, proveedores)
        self.alto = c["alto"]
        self.relleno = c.get("relleno", 1.5)
        self.rv, self.rh = c.get("recorte_v", 0.14), c.get("recorte_h", 0.2)
        self.lote = c.get("lote", 16)
        self.salidas = [o.name for o in self.cab.get_outputs()]

    def _a_linea(self, g: Image.Image):
        rot = False
        if self.cfg.get("enderezar_vertical") and g.size[1] > 1.3 * g.size[0]:
            g = g.transpose(Image.Transpose.ROTATE_90); rot = True
        w0 = g.size[0]
        g2 = I.recortar_fijo(g, self.rv, self.rh, self.cfg.get("recorte_trunc", False))
        dx = (w0 - g2.size[0]) / 2
        if self.cfg.get("normalizar"):
            g2 = I.normalizar_linea(g2)
        return g2, dx, rot

    def leer(self, recortes: list[Image.Image]) -> list[LineaLeida]:
        A = self.alto
        pasos = [self._a_linea(I.gris(g)) for g in recortes]
        tras = [p[0] for p in pasos]
        mc = self.cfg.get("max_cols", 160 if not self.cfg.get("prep2") else 240)
        prep = [I.preparar2(g, A, mc) if self.cfg.get("prep2") else I.preparar(g, A, mc) for g in tras]
        orden = sorted(range(len(prep)), key=lambda i: tras[i].size[0] / max(1, tras[i].size[1]))
        out: list = [None] * len(prep)
        for k in range(0, len(orden), self.lote):
            idx = orden[k : k + self.lote]
            W = max(prep[i].size[0] for i in idx)
            if self.relleno > 1:
                W = max(W, int(np.ceil(W * self.relleno / 16)) * 16)
            pix = np.zeros((len(idx), A, W), np.float32)
            cols = np.zeros(len(idx), np.int64)
            for j, i in enumerate(idx):
                a = np.asarray(prep[i], np.float32)
                pix[j, :, :] = I.fondo_derecho(a) / 255.0
                pix[j, :, : a.shape[1]] = a / 255.0
                cols[j] = a.shape[1] // 16
            est = self.cod.run(["estados"], {"pixeles": pix})[0]
            r = dict(zip(self.salidas, self.cab.run(None, {"estados": est, "columnas": cols})))
            for j, i in enumerate(idx):
                t = int(r["largos"][j])
                if "topv" in r:  # cabeza con top-k (alfabeto grande): el primero es el argmax
                    v, ii = r["topv"][j, :t], r["topi"][j, :t]
                    lp = np.full((t, len(self.chars) + 1), -40.0, np.float32)
                    np.put_along_axis(lp, ii.astype(np.int64), v, axis=1)
                else:
                    lp = r["logp"][j, :t]
                ll = voraz(lp, self.chars)
                # fotograma -> px del recorte: 4 fotogramas por columna de 16 px a alto A
                g = tras[i]
                esc = g.size[1] / A if not self.cfg.get("prep2") else g.size[0] / max(16, min(round(g.size[0] * A / max(1, g.size[1])), mc * 16))
                ll.xs = [pasos[i][1] + (x + 0.5) * 4 * esc for x in ll.xs]
                if self.cfg.get("bidi"):
                    ll.texto = a_logico(ll.texto)
                out[i] = ll
        return out


def a_logico(t: str) -> str:
    """Orden visual (de izquierda a derecha) -> orden lógico, para líneas con escritura de derecha a
    izquierda (aproximación: si predomina RTL, se invierte la línea y se reinvierten los tramos LTR)."""
    rtl = sum(unicodedata.bidirectional(c) in ("R", "AL") for c in t)
    ltr = sum(unicodedata.bidirectional(c) == "L" for c in t)
    if rtl <= ltr:
        return t
    import re

    s = t[::-1]
    return re.sub(r"[A-Za-z0-9][A-Za-z0-9 .,:;\-/]*[A-Za-z0-9]|[A-Za-z0-9]", lambda m: m.group(0)[::-1], s)


class LectorMini:
    """eg2-mini: mini.onnx (entradas x [B,1,alto,W] mapa de tinta, anchos [B]; salida logits [B,T,C])."""

    def __init__(self, carpeta: str, hilos: int = 0, proveedores: list | None = None):
        self.cfg = json.load(open(os.path.join(carpeta, "config.json")))
        self.chars = self.cfg["chars"]
        fich = self.cfg.get("fichero", "mini.onnx")
        if entorno("MINI_INT8", "0") == "1" and os.path.exists(os.path.join(carpeta, "mini_int8.onnx")):
            fich = "mini_int8.onnx"  # 8 bits solo con AMANUENSIS_MINI_INT8=1: la cuantización dinámica pierde mucho (CER 0,28 -> 1,50)
        self.s = sesion(os.path.join(carpeta, fich), hilos, proveedores)
        self.alto, self.esc_h = self.cfg["alto"], self.cfg.get("escala_h", 0.8)
        self.rv, self.rh = self.cfg.get("recorte_v", 0.06), self.cfg.get("recorte_h", 0.2)
        self.max_cols = self.cfg.get("max_cols", 160)

    def leer(self, recortes: list[Image.Image], lote: int = 64, max_area: int = 64 * 1600) -> list[LineaLeida]:
        A = self.alto
        prep = []
        for g in recortes:
            g = I.gris(g)
            w0 = g.size[0]
            g2 = I.normalizar_linea(I.recortar_fijo(g, self.rv, self.rh))
            dx = (w0 - g2.size[0]) / 2
            p = I.preparar2(g2, A, self.max_cols)
            nw = max(8, int(round(p.size[0] * self.esc_h)))
            p = p.resize((nw, A), Image.BILINEAR)
            prep.append((1.0 - np.asarray(p, np.float32) / 255.0, dx, g2.size[0] / max(1, nw)))
        orden = sorted(range(len(prep)), key=lambda i: prep[i][0].shape[1])
        out: list = [None] * len(prep)
        grupos, act = [], []
        for i in orden:  # lotes por ancho parecido con tope de área
            w = prep[i][0].shape[1]
            if act and (len(act) >= lote or (len(act) + 1) * w > max_area or w > 1.3 * max(64, prep[act[0]][0].shape[1])):
                grupos.append(act); act = []
            act.append(i)
        if act:
            grupos.append(act)
        for idx in grupos:
            W = -(-max(prep[i][0].shape[1] for i in idx) // 32) * 32
            x = np.zeros((len(idx), 1, A, W), np.float32)
            an = np.zeros(len(idx), np.int64)
            for j, i in enumerate(idx):
                a = prep[i][0]
                x[j, 0, :, : a.shape[1]] = a
                an[j] = a.shape[1]
            z = self.s.run(None, {"x": x, "anchos": an})
            logits, masc = z[0], z[1]
            for j, i in enumerate(idx):
                t = int(masc[j].sum())
                ll = voraz(logits[j, :t].astype(np.float32), self.chars, logits=True)
                ll.xs = [prep[i][1] + (f + 0.5) * 4 * prep[i][2] for f in ll.xs]
                out[i] = ll
        return out
