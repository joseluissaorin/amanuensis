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
    confs: list = None  # probabilidad de cada carácter (pico de la CTC)
    lp: object = None  # log-probabilidades [T, C] de la CTC (para la corrección léxica), si se guardan
    fotogramas: list = None  # fotograma de cada carácter (índices en lp)


def sesgo_blanco(arrojo: float) -> float:
    """«Arrojo» del lector (0-1; 0,5 = neutro) -> sesgo sumado al blanco de la CTC antes del argmax.
    Más arrojo = el blanco pesa menos y el lector escribe lo que apenas ve (escaneos muy malos); menos arrojo =
    más prudente (calla lo dudoso). Sumar al logit o a la log-probabilidad da el mismo argmax."""
    return (0.5 - min(1.0, max(0.0, float(arrojo)))) * 4.0


# signos «tímidos»: el lector los ve pero el blanco les gana por poco (una comilla que falta tiene un 10-45 % de
# probabilidad en su hueco; donde no hay ninguna, nada). Se escriben si pasan de SIGNOS_UMBRAL.
SIGNOS_TIMIDOS = "‘’'“”\"«»"
SIGNOS_UMBRAL = float(os.environ.get("AMANUENSIS_SIGNOS", "0.08"))


def _signos_timidos(lp: np.ndarray, ids: np.ndarray, chars: list[str], umbral: float) -> tuple[np.ndarray, dict]:
    """Fotogramas donde gana el blanco pero un signo tímido pasa del umbral: se le da el fotograma (el más probable de
    cada racha, para no escribir dos). Devuelve los ids cambiados y {fotograma: probabilidad}."""
    cand = [i + 1 for i, c in enumerate(chars) if c in SIGNOS_TIMIDOS]
    if not cand or umbral <= 0:
        return ids, {}
    p = np.exp(lp[:, cand]) if lp.max() <= 0 else None
    if p is None:  # logits: a probabilidades
        z = lp - lp.max(-1, keepdims=True)
        p = (np.exp(z) / np.exp(z).sum(-1, keepdims=True))[:, cand]
    mejor = p.argmax(1)
    pm = p[np.arange(len(p)), mejor]
    ids = ids.copy()
    nuevas = {}
    t, T = 0, len(ids)
    while t < T:
        if ids[t] == 0 and pm[t] >= umbral:
            u = t
            while u + 1 < T and ids[u + 1] == 0 and pm[u + 1] >= umbral:
                u += 1
            k = t + int(np.argmax(pm[t:u + 1]))
            c = cand[mejor[k]]
            # ni junto a otro signo tímido ya escrito («cita’’»): el símbolo más cercano a cada lado no puede serlo
            antes = next((ids[j] for j in range(k - 1, max(-1, k - 8), -1) if ids[j] != 0), 0)
            despues = next((ids[j] for j in range(k + 1, min(T, k + 8)) if ids[j] != 0), 0)
            vecino = antes in cand or despues in cand
            if not vecino:
                ids[k] = c
                nuevas[k] = float(pm[k])
            t = u + 1
        else:
            t += 1
    return ids, nuevas


def voraz(lp: np.ndarray, chars: list[str], logits: bool = False, sesgo: float = 0.0, signos: float = 0.0) -> LineaLeida:
    """lp [T, C] (log-probabilidades, o logits si logits=True) -> texto, confianza media de los picos,
    x por carácter (en fotogramas) y probabilidad de cada carácter. `sesgo` se suma al blanco (clase 0) solo para
    decidir; las probabilidades que se devuelven son las del modelo, sin sesgo."""
    if sesgo:
        q = lp.copy()
        q[:, 0] += sesgo
        ids = q.argmax(-1)
    else:
        ids = lp.argmax(-1)
    nuevas = {}
    if signos > 0:
        ids, nuevas = _signos_timidos(lp, ids, chars, signos)
    if logits:
        z = lp - lp.max(-1, keepdims=True)
        p = 1.0 / np.exp(z).sum(-1)
    else:
        p = np.exp(lp.max(-1))
    if nuevas:  # la probabilidad del signo tímido, no la del blanco que ganaba
        p = p.copy()
        for k, v in nuevas.items():
            p[k] = v
    if sesgo:  # probabilidad del símbolo elegido (que con sesgo puede no ser el máximo)
        if logits:
            z = lp - lp.max(-1, keepdims=True)
            p = np.exp(np.take_along_axis(z, ids[:, None], 1)[:, 0]) / np.exp(z).sum(-1)
        else:
            p = np.exp(np.take_along_axis(lp, ids[:, None], 1)[:, 0])
    out, xs, confs, prev = [], [], [], 0
    for t, i in enumerate(ids):
        if i != prev and i != 0:
            out.append(chars[i - 1]); xs.append(t); confs.append(float(p[t]))
        elif i == prev and i != 0 and confs:
            confs[-1] = max(confs[-1], float(p[t]))
        prev = i
    return LineaLeida("".join(out), float(np.mean(confs)) if confs else 0.0, xs, confs)


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

    def leer(self, recortes: list[Image.Image], sesgo: float = 0.0) -> list[LineaLeida]:
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
                ll = voraz(lp, self.chars, sesgo=sesgo)
                # fotograma -> px del recorte: 4 fotogramas por columna de 16 px a alto A
                g = tras[i]
                esc = g.size[1] / A if not self.cfg.get("prep2") else g.size[0] / max(16, min(round(g.size[0] * A / max(1, g.size[1])), mc * 16))
                ll.xs = [pasos[i][1] + (x + 0.5) * 4 * esc for x in ll.xs]
                if self.cfg.get("bidi"):
                    t2 = a_logico(ll.texto)
                    if t2 != ll.texto:  # orden lógico: x y confianzas siguen al carácter
                        perm = permutacion_logica(ll.texto)
                        ll.xs = [ll.xs[k] for k in perm]; ll.confs = [ll.confs[k] for k in perm]
                    ll.texto = t2
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


def permutacion_logica(t: str) -> list:
    """Índices de t en el orden de a_logico(t)."""
    import re

    idx = list(range(len(t)))
    s, ri = t[::-1], idx[::-1]
    out, k = [], 0
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9 .,:;\-/]*[A-Za-z0-9]|[A-Za-z0-9]", s):
        out += ri[k:m.start()] + ri[m.start():m.end()][::-1]
        k = m.end()
    return out + ri[k:]


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

    def _preparar(self, recortes: list[Image.Image]) -> list:
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
        return prep

    def _logits(self, arrays: list[np.ndarray], lote: int = 64, max_area: int = 64 * 1600) -> list[np.ndarray]:
        """Logits [T, C] de cada imagen preparada (lotes por ancho parecido con tope de área)."""
        A = self.alto
        orden = sorted(range(len(arrays)), key=lambda i: arrays[i].shape[1])
        out: list = [None] * len(arrays)
        grupos, act = [], []
        for i in orden:
            w = arrays[i].shape[1]
            if act and (len(act) >= lote or (len(act) + 1) * w > max_area or w > 1.3 * max(64, arrays[act[0]].shape[1])):
                grupos.append(act); act = []
            act.append(i)
        if act:
            grupos.append(act)
        for idx in grupos:
            W = -(-max(arrays[i].shape[1] for i in idx) // 32) * 32
            x = np.zeros((len(idx), 1, A, W), np.float32)
            an = np.zeros(len(idx), np.int64)
            for j, i in enumerate(idx):
                a = arrays[i]
                x[j, 0, :, : a.shape[1]] = a
                an[j] = a.shape[1]
            z = self.s.run(None, {"x": x, "anchos": an})
            logits, masc = z[0], z[1]
            for j, i in enumerate(idx):
                out[i] = logits[j, : int(masc[j].sum())].astype(np.float32)
        return out

    @staticmethod
    def _logsoftmax(lg: np.ndarray) -> np.ndarray:
        z = lg - lg.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    def _decodificar(self, lp: np.ndarray, dx: float, esc: float, sesgo: float) -> LineaLeida:
        ll = voraz(lp, self.chars, logits=False, sesgo=sesgo, signos=SIGNOS_UMBRAL)
        ll.fotogramas = list(ll.xs)
        ll.lp = lp
        ll.xs = [dx + (f + 0.5) * 4 * esc for f in ll.xs]
        return ll

    def leer(self, recortes: list[Image.Image], lote: int = 64, max_area: int = 64 * 1600, sesgo: float = 0.0) -> list[LineaLeida]:
        prep = self._preparar(recortes)
        lgs = self._logits([a for a, _, _ in prep], lote, max_area)
        return [self._decodificar(self._logsoftmax(lg), dx, esc, sesgo) for lg, (_, dx, esc) in zip(lgs, prep)]
