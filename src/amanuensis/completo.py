"""Modelo completo de Amanuensis: la cadena de página entera sin PyTorch (ONNX Runtime en CPU o CUDA).

    from amanuensis.completo import Completo
    c = Completo(carpeta)                       # carpeta con completo.json y los ONNX
    res = c.leer([PIL.Image, ...])

Piezas (todas en `carpeta`, ver completo.json):
  1. renglones de alta resolución (LineasAR) con los rasgos de EmbeddingGemma 2 de la página a 960 px;
  2. maquetación v4 (zonas, tipos de línea, tablas y orden de lectura aprendido) con su propio codificador (LoRA fundido);
  3. recorte enderezado por la línea base (cv2.remap), con máscara del polígono, papel local y contraste normalizado;
  4. lector CTC grande (codificador + cabeza top-k) y selector por escritura: si el multilingüe lee en una muestra de
     6 líneas más de la mitad de letras no latinas ni griegas, la página va al multilingüe.
Mismo resultado que pagina/cadena.py CadenaPagina del repositorio de entrenamiento, con dos campos más por línea:
«confs» (probabilidad de cada carácter) y «xs» (x de cada carácter en píxeles de la página).

`arrojo` ∈ [0, 1] (0,5 = neutro): sesgo del símbolo en blanco de la CTC, b = (0,5 − arrojo) × 4, sumado a su
log-probabilidad antes del argmax. Más arrojo = el lector se calla menos (escaneos muy malos); menos = más prudente.
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
import unicodedata

import numpy as np
from PIL import Image

from . import completo_cpu as CC

TEXTO_Z = ("MainZone", "MarginTextZone", "RunningTitleZone", "NumberingZone", "Text", "List-item", "Section-header", "Title",
           "Caption", "Footnote", "Page-header", "Page-footer")
LATINAS = {"LATIN", "GREEK"}


def _sesion(ruta, hilos=0, proveedores=None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if hilos:
        so.intra_op_num_threads = hilos
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False  # formas variables: los patrones de memoria solo acumulan
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(ruta, so, providers=proveedores or ["CPUExecutionProvider"])


def sesgo_blanco(arrojo: float) -> float:
    return (0.5 - float(arrojo)) * 4.0


def escritura(ch: str) -> str:
    try:
        return unicodedata.name(ch).split(" ")[0]
    except ValueError:
        return "?"


def a_logico_idx(t: str):
    """Como amanuensis.lector.a_logico, pero devuelve también la permutación de índices (para confs y xs)."""
    rtl = sum(unicodedata.bidirectional(c) in ("R", "AL") for c in t)
    ltr = sum(unicodedata.bidirectional(c) == "L" for c in t)
    idx = list(range(len(t)))
    if rtl <= ltr:
        return t, idx
    import re

    s = t[::-1]
    ri = idx[::-1]
    out_t, out_i, k = [], [], 0
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9 .,:;\-/]*[A-Za-z0-9]|[A-Za-z0-9]", s):
        out_t.append(s[k:m.start()]); out_i += ri[k:m.start()]
        out_t.append(m.group(0)[::-1]); out_i += ri[m.start():m.end()][::-1]
        k = m.end()
    out_t.append(s[k:]); out_i += ri[k:]
    return "".join(out_t), out_i


def a_norm(v, n_celdas):
    return ((v - 8.0) / max(1.0, n_celdas * 16 - 16.0)) * 2 - 1


def media_cajas(f: np.ndarray, cajas: np.ndarray) -> np.ndarray:
    """f [C,H,W], cajas [n,4] en celdas -> media de f dentro de cada caja [n,C] (al menos una celda)."""
    C, H, W = f.shape
    ii = np.zeros((C, H + 1, W + 1), np.float64)
    ii[:, 1:, 1:] = f.astype(np.float64).cumsum(1).cumsum(2)
    x0 = np.clip(np.floor(cajas[:, 0]), 0, W - 1).astype(int)
    y0 = np.clip(np.floor(cajas[:, 1]), 0, H - 1).astype(int)
    x1 = np.maximum(np.clip(np.ceil(cajas[:, 2]), 1, W).astype(int), x0 + 1)
    y1 = np.maximum(np.clip(np.ceil(cajas[:, 3]), 1, H).astype(int), y0 + 1)
    s = ii[:, y1, x1] - ii[:, y0, x1] - ii[:, y1, x0] + ii[:, y0, x0]
    return (s / ((x1 - x0) * (y1 - y0))).T.astype(np.float32)


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x * (1.0 / math.sqrt(2.0))))


def _erf(x):
    # aproximación de Abramowitz-Stegun 7.1.26 mejorada (error < 1,5e-7)
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


class LectorCTC:
    """Lector en el formato de la demo web (config.json + codificador_<var>.onnx + cabeza_<nombre>_<var>.onnx) que lee
    recortes ya enderezados al alto del lector ([alto, w16] en 0-255, papel claro)."""

    def __init__(self, carpeta, variantes=("q8", "q8_b32", "fp32"), hilos=0, proveedores=None, cabeza="literal"):
        self.cfg = json.load(open(os.path.join(carpeta, "config.json")))
        c = self.cfg
        self.chars = c["chars"] if isinstance(c["chars"], list) else json.loads(c["chars"])
        self.alto, self.max_cols = c["alto"], c.get("max_cols", 240)
        nombre_cab = (c.get("cabezas") or {}).get(cabeza, "cabeza")
        var = c.get("variante_completo")
        cands = [var] if var else list(variantes)
        for v in cands:
            rc, rh = os.path.join(carpeta, f"codificador_{v}.onnx"), os.path.join(carpeta, f"{nombre_cab}_{v}.onnx")
            if os.path.exists(rc) and os.path.exists(rh):
                break
        else:
            raise FileNotFoundError(f"{carpeta}: no hay codificador/cabeza en {cands}")
        self.variante = v
        self.cod = _sesion(rc, hilos, proveedores)
        self.cab = _sesion(rh, hilos, proveedores)
        self.salidas = [o.name for o in self.cab.get_outputs()]

    def leer(self, recortes, presupuesto=8 * 240, sesgo=0.0):
        """-> [(texto, conf, confs, frames)] (frames: fotograma del pico de cada carácter)."""
        if not recortes:
            return []
        orden = sorted(range(len(recortes)), key=lambda i: recortes[i].shape[1])
        out = [None] * len(recortes)
        k = 0
        while k < len(orden):
            j = k
            while j < len(orden) and (j - k + 1) * (recortes[orden[j]].shape[1] // 16) <= presupuesto:
                j += 1
            j = max(j, k + 1)
            idx = orden[k:j]
            W = max(recortes[i].shape[1] for i in idx)
            pix = np.ones((len(idx), self.alto, W), np.float32)
            cols = np.zeros(len(idx), np.int64)
            for r, i in enumerate(idx):
                w = recortes[i].shape[1]
                pix[r, :, :w] = recortes[i] / 255.0
                cols[r] = w // 16
            est = self.cod.run(["estados"], {"pixeles": pix})[0]
            res = dict(zip(self.salidas, self.cab.run(None, {"estados": est, "columnas": cols})))
            for r, i in enumerate(idx):
                t = int(res["largos"][r])
                if "topv" in res:
                    v, ii = res["topv"][r, :t].astype(np.float32), res["topi"][r, :t].astype(np.int64)
                else:  # matriz entera: se queda con las 16 mejores (mismo resultado para el argmax con sesgo)
                    lp = res["logp"][r, :t]
                    ii = np.argsort(-lp, -1)[:, :16]
                    v = np.take_along_axis(lp, ii, -1)
                out[i] = self._decodificar(v, ii, sesgo)
            k = j
        return out

    def _decodificar(self, v, ii, sesgo):
        T = v.shape[0]
        if T == 0:
            return "", 0.0, [], []
        conf = float(np.exp(v.max(-1)).mean())
        if sesgo:
            v = v + np.where(ii == 0, sesgo, 0.0)
            if sesgo > 0:  # el blanco fuera del top-k: su log-prob es <= la última; con sesgo positivo puede ganar
                sin_b = ~(ii == 0).any(-1)
                vb = v[:, -1] + sesgo
                am = v.argmax(-1)
                ids = np.where(sin_b & (vb > v[np.arange(T), am]), 0, ii[np.arange(T), am])
                pk = np.exp(v[np.arange(T), am] - np.where(ii[np.arange(T), am] == 0, sesgo, 0.0))
            else:
                am = v.argmax(-1)
                ids = ii[np.arange(T), am]
                pk = np.exp(v[np.arange(T), am] - np.where(ids == 0, sesgo, 0.0))
        else:
            am = v.argmax(-1)
            ids = ii[np.arange(T), am]
            pk = np.exp(v[np.arange(T), am])
        txt, confs, frames, prev = [], [], [], 0
        for t in range(T):
            i = int(ids[t])
            if i != 0 and i != prev:
                txt.append(self.chars[i - 1]); confs.append(float(pk[t])); frames.append(t)
            elif i != 0 and i == prev:
                confs[-1] = max(confs[-1], float(pk[t]))
            prev = i
        return "".join(txt), conf, confs, frames


class Completo:
    def __init__(self, carpeta, proveedores=None, hilos=0, arrojo=0.5, presupuesto=0, coreml="auto"):
        """coreml: «auto» (en macOS con coremltools y los paquetes de completo.json → «coreml», el lector grande y la
        maquetación van en la GPU del Mac), «si» o «no»."""
        self.dir = carpeta
        cfg = self.cfg = json.load(open(os.path.join(carpeta, "completo.json")))
        p = lambda k: os.path.join(carpeta, cfg["piezas"][k])
        self._p, self._hilos, self._prov = p, hilos, proveedores
        self.s_base = _sesion(p("cod_base"), hilos, proveedores)
        self.s_lin = _sesion(p("lineas"), hilos, proveedores)
        self.s_ord = _sesion(p("orden"), hilos, proveedores)
        self._s_mq = self._s_maq = None
        op = np.load(p("orden_par"))
        self.op = {k: op[k].astype(np.float32) for k in op.files}
        vl = tuple(cfg.get("variantes_lector", ("q8", "q8_b32", "fp32")))
        cm = cfg.get("coreml") or {}
        usar_cm = False
        if coreml != "no" and cm and not (proveedores and any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in proveedores)):
            from . import completo_coreml as CM

            usar_cm = CM.disponible() and all(os.path.isdir(os.path.join(carpeta, v)) for v in cm.values())
            if coreml == "si" and not usar_cm:
                raise RuntimeError("Core ML no disponible (macOS 15+, coremltools y los paquetes de completo.json)")
        self.cm_maq = None
        if usar_cm:
            self.lec = CM.LectorCTCCoreML(os.path.join(carpeta, cm["lector"]))
            self.cm_maq = CM.MaquetaCoreML(os.path.join(carpeta, cm["maqueta"])) if "maqueta" in cm else None
        else:
            self.lec = LectorCTC(self._ruta(cfg["lector"]), vl, hilos=hilos, proveedores=proveedores)
        ml = cfg.get("lector_ml")
        self.ml = None
        if ml:
            try:
                self.ml = LectorCTC(self._ruta(ml), vl, hilos=hilos, proveedores=proveedores)
            except FileNotFoundError:
                self.ml = None
        self.zonas, self.tipos_linea = cfg["zonas"], cfg["lineas"]
        self.nz_texto = {self.zonas.index(c) for c in TEXTO_Z if c in self.zonas}
        self.tabla_idx = {self.zonas.index(c) for c in ("Table", "TableZone") if c in self.zonas}
        self.altos = sorted({x.alto for x in (self.lec, self.ml) if x is not None})
        self.lado, self.umbral = cfg.get("lado", 2048), cfg.get("umbral", 0.5)
        self.umbral_ml = cfg.get("umbral_ml", 0.5)
        self.paso = cfg.get("paso_lineas", 2)
        self.arrojo = arrojo
        self.presupuesto = presupuesto or cfg.get("presupuesto", 8 * 240)
        self.crono = {}
        # rama densa (páginas con >= umbral renglones: otra maquetación, orden jerárquico y renglones por teselas)
        dz = cfg.get("densa") or {}
        self.densa = dz if dz and all(os.path.exists(os.path.join(carpeta, dz[k])) for k in ("cod_mq", "maqueta", "orden", "orden_par")) else None
        self._dz = None
        if os.environ.get("AMANUENSIS_DENSA", "1") == "0":
            self.densa = None

    def _en_gpu(self):
        return bool(self._prov) and any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in self._prov)

    def _sesiones_densa(self):
        if self._dz is None:
            j = lambda k: os.path.join(self.dir, self.densa[k])
            op = np.load(j("orden_par"))
            self._dz = {"mq": _sesion(j("cod_mq"), self._hilos, self._prov), "maq": _sesion(j("maqueta"), self._hilos, self._prov),
                        "ord": _sesion(j("orden"), self._hilos, self._prov), "op": {k: op[k].astype(np.float32) for k in op.files}}
        return self._dz

    def _ruta(self, r):
        if os.path.isabs(r) or os.path.exists(os.path.join(self.dir, r)):
            return os.path.join(self.dir, r)
        from . import modelos  # nombre del catálogo (p. ej. «multilingue»)

        return modelos.ruta(r)

    def _t(self, k, t0):
        self.crono[k] = self.crono.get(k, 0.0) + time.time() - t0

    # ---------------------------------------------------------------- una página
    def _preparar(self, img):
        img = img.convert("L")
        W0, H0 = img.size
        eo = min(1.0, 5000 / max(W0, H0))
        orig = img if eo == 1 else img.resize((round(W0 * eo), round(H0 * eo)), Image.LANCZOS)
        e = self.lado / max(W0, H0)
        W, H = round(W0 * e), round(H0 * e)
        Wc, Hc = int(math.ceil(W / 32)) * 32, int(math.ceil(H / 32)) * 32
        lz = Image.new("L", (Wc, Hc), 255)
        lz.paste(img.resize((W, H), Image.BILINEAR), (0, 0))
        e9 = 960 / Wc
        if Hc * e9 > 1920:
            e9 = 1920 / Hc
        w9, h9 = max(16, round(Wc * e9)), max(16, round(Hc * e9))
        W9, H9 = int(math.ceil(w9 / 16)) * 16, int(math.ceil(h9 / 16)) * 16
        l9 = Image.new("L", (W9, H9), 255)
        l9.paste(lz.resize((w9, h9), Image.BILINEAR), (0, 0))
        em = 960 / W0
        Hm = int(math.ceil(H0 * em / 48)) * 48
        lm = Image.new("L", (960, Hm), 255)
        lm.paste(img.resize((960, max(1, int(round(H0 * em)))), Image.BILINEAR), (0, 0))
        a = lambda im: np.asarray(im, np.uint8)
        return {"orig": a(orig), "eo": eo, "lz": a(lz), "e": e, "l9": a(l9), "e9": e9, "lm": a(lm), "em": em,
                "W0": W0, "H0": H0, "Wv": W, "Hv": H}

    def _mapas(self, x, densa=None):
        """-> (trabajo para decodificar, estado para el orden). densa: sesiones de la rama densa (maquetación propia, sin
        Core ML) y renglones por teselas si la página los pide."""
        t0 = time.time()
        l9 = x["l9"].astype(np.float32)[None] / 255.0
        est9 = self.s_base.run(["estados"], {"pixeles": l9})[0]
        gh, gw = l9.shape[1] // 16, l9.shape[2] // 16
        G = x["lz"].astype(np.float32)[None, None] / 255.0
        Hb, Wb = G.shape[-2:]
        h16, w16 = Hb // 16, Wb // 16
        e9 = x["e9"]
        c = [a_norm(0, gw), a_norm(0, gh), a_norm(Wb * e9, gw), a_norm(Hb * e9, gh)]
        gx = c[0] + (c[2] - c[0]) * np.linspace(0, 1, w16, dtype=np.float32)
        gy = c[1] + (c[3] - c[1]) * np.linspace(0, 1, h16, dtype=np.float32)
        rej = np.stack([np.broadcast_to(gx[None, :], (h16, w16)), np.broadcast_to(gy[:, None], (h16, w16))], -1)[None].astype(np.float32)
        if "_lin" in x:  # la rama densa reutiliza los renglones de la pasada base
            nuc, lh = x["_lin"]
        else:
            nuc, lh = self.s_lin.run(None, {"gris": G, "estados": est9, "celdas": np.zeros((gh, gw), np.float32), "rejilla": rej})
            x["_lin"], x["_est9"] = (nuc, lh), (est9, gh, gw)
        self._t("lineas", t0)
        t0 = time.time()
        lm = x["lm"].astype(np.float32) / 255.0
        if densa is not None:
            estm = densa["mq"].run(["estados"], {"pixeles": lm[None]})[0]
            zn, zr, ln, tb, y4, h16m = densa["maq"].run(None, {"gris": lm[None, None], "estados": estm})
        elif self.cm_maq is not None and self.cm_maq.cabe(lm.shape[0]):
            zn, zr, ln, tb, y4, h16m = self.cm_maq.mapas(lm)
        else:
            if self._s_mq is None:
                self._s_mq = _sesion(self._p("cod_mq"), self._hilos, self._prov)
                self._s_maq = _sesion(self._p("maqueta"), self._hilos, self._prov)
            estm = self._s_mq.run(["estados"], {"pixeles": lm[None]})[0]
            zn, zr, ln, tb, y4, h16m = self._s_maq.run(None, {"gris": lm[None, None], "estados": estm})
        self._t("maqueta", t0)
        hv, wv = int(math.ceil(x["Hv"] / self.paso)), int(math.ceil(x["Wv"] / self.paso))
        h4 = int(math.ceil(x["H0"] * x["em"] / 4))
        nl, ll, el = nuc[0, 0, :hv, :wv], lh[0, :, :hv, :wv], x["e"]
        if densa is not None and self.densa.get("teselas", "auto") == "auto" and os.environ.get("AMANUENSIS_TESELAS", "1") != "0":
            et = self._escala_teselas(x, nl, ll)
            if et is not None:
                t0 = time.time()
                nl, ll, el = self._lineas_por_teselas(x, et)
                self._t("teselas", t0)
        trab = {"nuc": nl, "lh": ll, "esc_l": el, "paso": self.paso, "umbral": self.umbral,
                "zn": zn[0, :, :h4], "zr": zr[0, :, :h4], "tb": tb[0, :, :h4], "esc_m": x["em"],
                "altos": self.altos, "max_cols": self.lec.max_cols, "nz_texto": self.nz_texto, "tabla_idx": self.tabla_idx}
        estado = {"h16": h16m[0], "y4": y4[0], "ln": ln[0], "Hc": lm.shape[0]}
        return trab, estado

    # ---------------------------------------------------------------- teselas (rama densa)
    def _escala_teselas(self, x, nuc, lh, altura_min=16.0, altura_objetivo=32.0, lado_max=12000, lado_original=5000):
        """Como pagina/cadena.py (--teselas auto): si la mediana del alto de renglón a lado `lado` es < altura_min px, la
        escala (respecto al original) que la deja en altura_objetivo; None si basta la pasada normal."""
        nativo = min(1.0, lado_original / max(x["W0"], x["H0"]))
        m = 1 / (1 + np.exp(-nuc.astype(np.float32))) > self.umbral
        if int(m.sum()) < 50:
            return None
        alt = (np.expm1(lh[0].astype(np.float32)) + np.expm1(lh[1].astype(np.float32)))[m] * self.paso
        hmed = float(np.median(alt))
        if hmed >= altura_min:
            return None
        et = min(x["e"] * altura_objetivo / max(hmed, 1.0), nativo, lado_max / max(x["W0"], x["H0"]))
        return et if et > x["e"] * 1.2 else None

    def _lineas_por_teselas(self, x, et, T=2048):
        """LineasAR por teselas de T px a escala et con solape de 1/4 y los rasgos de EG2 de la página entera (la rejilla
        de cada tesela apunta a su trozo del mapa de EG2); los mapas se promedian con una ventana que pesa poco en los
        bordes. -> (nuc, lh, et)."""
        est9, gh, gw = x["_est9"]
        orig = Image.fromarray(x["orig"])
        W, H = max(1, round(x["W0"] * et)), max(1, round(x["H0"] * et))
        big = np.asarray(orig.resize((W, H), Image.BILINEAR if et >= x["eo"] else Image.LANCZOS), np.float32) / 255.0
        Wc, Hc = max(T, int(math.ceil(W / 32)) * 32), max(T, int(math.ceil(H / 32)) * 32)
        lz = np.ones((Hc, Wc), np.float32)
        lz[:H, :W] = big
        del big
        pso = T - T // 4
        xs = list(range(0, max(1, Wc - T) + 1, pso))
        if xs[-1] + T < Wc:
            xs.append(Wc - T)
        ys = list(range(0, max(1, Hc - T) + 1, pso))
        if ys[-1] + T < Hc:
            ys.append(Hc - T)
        p = self.paso
        tp = T // p
        v = np.minimum(np.arange(tp) + 1, tp - np.arange(tp)).astype(np.float32)
        v = np.clip(v / (tp / 8), None, 1.0)
        ven = v[:, None] * v[None, :]
        nuc = np.zeros((Hc // p, Wc // p), np.float32)
        lh = np.zeros((2, Hc // p, Wc // p), np.float32)
        peso = np.zeros((Hc // p, Wc // p), np.float32)
        k = (x["e"] / et) * x["e9"]  # px a escala et -> px de la imagen de EG2
        n16 = T // 16
        celdas = np.zeros((gh, gw), np.float32)
        for yy in ys:
            for xx in xs:
                c = [a_norm(xx * k, gw), a_norm(yy * k, gh), a_norm((xx + T) * k, gw), a_norm((yy + T) * k, gh)]
                gx = c[0] + (c[2] - c[0]) * np.linspace(0, 1, n16, dtype=np.float32)
                gy = c[1] + (c[3] - c[1]) * np.linspace(0, 1, n16, dtype=np.float32)
                rej = np.stack([np.broadcast_to(gx[None, :], (n16, n16)), np.broadcast_to(gy[:, None], (n16, n16))], -1)[None].astype(np.float32)
                G = lz[None, None, yy: yy + T, xx: xx + T]
                n_, l_ = self.s_lin.run(None, {"gris": np.ascontiguousarray(G), "estados": est9, "celdas": celdas, "rejilla": rej})
                y2, x2 = yy // p, xx // p
                nuc[y2: y2 + tp, x2: x2 + tp] += n_[0, 0] * ven
                lh[:, y2: y2 + tp, x2: x2 + tp] += l_[0] * ven
                peso[y2: y2 + tp, x2: x2 + tp] += ven
        peso = np.maximum(peso, 1e-6)
        hv, wv = int(math.ceil(H / p)), int(math.ceil(W / p))
        return (nuc / peso)[:hv, :wv], (lh / peso)[:, :hv, :wv], et

    # ---------------------------------------------------------------- orden de lectura
    def _rasgos(self, estado, x, cj, zb, zc, lt, s_ord):
        """Rasgos de ítems (líneas o bloques) como Maqueta.rasgos_items + o_in/o_tr -> (h, cajas normalizadas)."""
        n = cj.shape[0]
        W, H = 960.0, float(estado["Hc"])
        r16 = media_cajas(estado["h16"], cj / 16)
        r4 = media_cajas(estado["y4"], cj / 4)
        esc = np.array([W, H, W, H], np.float32)
        nc, nz = cj / esc, zb / esc
        geo = np.concatenate([nc, nc[:, 2:] - nc[:, :2], (nc[:, :2] + nc[:, 2:]) / 2, nz[:, :2], nz[:, 2:] - nz[:, :2]], 1)
        NZ, NL = len(self.zonas), len(self.tipos_linea)
        oz = np.zeros((n, NZ), np.float32)
        sel = zc >= 0
        oz[np.nonzero(sel)[0], zc[sel]] = 1
        ol = np.zeros((n, NL), np.float32)
        ol[np.arange(n), lt] = 1
        rel = np.concatenate([(nc[:, :2] - nz[:, :2]), (nz[:, 2:] - nc[:, 2:])], 1)
        oxy = CC.corte_xy(cj.tolist())
        rk = np.empty(n, np.float32)
        rk[np.array(oxy)] = np.arange(n, dtype=np.float32) / max(1, n - 1)
        xf = np.concatenate([r16, r4, geo, oz, ol, rel, rk[:, None]], 1).astype(np.float32)
        return s_ord.run(["h"], {"x": xf})[0], nc

    @staticmethod
    def _pares(h, nc, op):
        """Logits antisimétricos [n, n] de «i antes que j» (o_par por bloques de filas: memoria lineal)."""
        n = h.shape[0]
        w1, b1, w2, b2 = op["w1"], op["b1"], op["w2"], op["b2"]
        d = h.shape[1]
        A = h @ w1[:, :d].T + b1
        B = h @ w1[:, d:2 * d].T
        Cg = w1[:, 2 * d:]
        s = np.empty((n, n), np.float32)
        paso = max(1, 4_000_000 // (n * d))
        for r0 in range(0, n, paso):
            r1 = min(n, r0 + paso)
            ci, cjn = nc[r0:r1, None, :], nc[None, :, :]
            g = np.concatenate([np.broadcast_to(ci[..., :2] - cjn[..., :2], (r1 - r0, n, 2)),
                                np.broadcast_to(ci[..., 2:] - cjn[..., 2:], (r1 - r0, n, 2)),
                                np.broadcast_to(ci[..., 1:2] - cjn[..., 3:4], (r1 - r0, n, 1)),
                                np.broadcast_to(cjn[..., 1:2] - ci[..., 3:4], (r1 - r0, n, 1))], -1)
            z = A[r0:r1, None, :] + B[None, :, :] + g @ Cg.T
            s[r0:r1] = (_gelu(z) @ w2[0] + b2[0])
        return (s - s.T) / 2

    @staticmethod
    def _ordenar(lg):
        p = 1 / (1 + np.exp(-np.clip(lg, -60, 60)))
        np.fill_diagonal(p, 0)
        gana = (p > 0.5).sum(1) + 1e-3 * p.sum(1)
        return [int(i) for i in np.argsort(-gana, kind="stable")]

    def _orden_jerarquico(self, estado, x, lineas, s_ord, op):
        """maqueta/orden_jer.py en numpy: bloques = zonas (las líneas sin zona, por columnas), orden de bloques con la
        misma cabeza sobre ítems de bloque y luego líneas dentro de cada bloque; bloques de más de 600 líneas, en trozos."""
        n = len(lineas)
        em = x["em"]
        cj = np.array([[v * em for v in l["caja"]] for l in lineas], np.float32)
        zb = np.array([[v * em for v in l["zcaja"]] for l in lineas], np.float32)
        zc = np.array([l["zona"] for l in lineas])
        lt = media_cajas(estado["ln"][1:], cj / 4).argmax(1)
        h, nc = self._rasgos(estado, x, cj, zb, zc, lt, s_ord)
        ids = {}
        zid = [ids.setdefault(tuple(l["zcaja"]), len(ids)) if l["zona"] >= 0 else -1 for l in lineas]
        bl = CC.bloques_de(cj.tolist(), zid)
        if len(bl) > 1:
            cl = cj.tolist()
            bc = np.array([[min(cl[i][0] for i in b), min(cl[i][1] for i in b), max(cl[i][2] for i in b), max(cl[i][3] for i in b)] for b in bl], np.float32)
            bz = np.array([zc[b[0]] for b in bl])
            hb, ncb = self._rasgos(estado, x, bc, bc, bz, np.zeros(len(bl), np.int64), s_ord)
            ob = self._ordenar(self._pares(hb, ncb, op))
        else:
            ob = [0]
        out = []
        for k in ob:
            b = bl[k]
            if len(b) == 1:
                out += b
                continue
            o = self._ordenar(self._pares(h[b], nc[b], op))
            out += [b[j] for j in o]
        return out

    def _orden(self, estado, x, lineas):
        n = len(lineas)
        if n <= 1:
            return list(range(n))
        em = x["em"]
        W, H = 960.0, float(estado["Hc"])
        cj = np.array([[v * em for v in l["caja"]] for l in lineas], np.float32)
        zb = np.array([[v * em for v in l["zcaja"]] for l in lineas], np.float32)
        zc = np.array([l["zona"] for l in lineas])
        lt = media_cajas(estado["ln"][1:], cj / 4).argmax(1)
        r16 = media_cajas(estado["h16"], cj / 16)
        r4 = media_cajas(estado["y4"], cj / 4)
        esc = np.array([W, H, W, H], np.float32)
        nc, nz = cj / esc, zb / esc
        geo = np.concatenate([nc, nc[:, 2:] - nc[:, :2], (nc[:, :2] + nc[:, 2:]) / 2, nz[:, :2], nz[:, 2:] - nz[:, :2]], 1)
        NZ, NL = len(self.zonas), len(self.tipos_linea)
        oz = np.zeros((n, NZ), np.float32)
        sel = zc >= 0
        oz[np.nonzero(sel)[0], zc[sel]] = 1
        ol = np.zeros((n, NL), np.float32)
        ol[np.arange(n), lt] = 1
        rel = np.concatenate([(nc[:, :2] - nz[:, :2]), (nz[:, 2:] - nc[:, 2:])], 1)
        oxy = CC.corte_xy(cj.tolist())
        rk = np.empty(n, np.float32)
        rk[np.array(oxy)] = np.arange(n, dtype=np.float32) / max(1, n - 1)
        xf = np.concatenate([r16, r4, geo, oz, ol, rel, rk[:, None]], 1).astype(np.float32)
        h = self.s_ord.run(["h"], {"x": xf})[0]
        # o_par por bloques: Linear(2d+6 -> d) separado en h_i, h_j y la geometría del par
        w1, b1, w2, b2 = self.op["w1"], self.op["b1"], self.op["w2"], self.op["b2"]
        d = h.shape[1]
        A = h @ w1[:, :d].T + b1
        B = h @ w1[:, d:2 * d].T
        Cg = w1[:, 2 * d:]  # [d, 6]
        s = np.empty((n, n), np.float32)
        paso = max(1, 4_000_000 // (n * d))
        for r0 in range(0, n, paso):
            r1 = min(n, r0 + paso)
            ci, cjn = nc[r0:r1, None, :], nc[None, :, :]
            g = np.concatenate([np.broadcast_to(ci[..., :2] - cjn[..., :2], (r1 - r0, n, 2)),
                                np.broadcast_to(ci[..., 2:] - cjn[..., 2:], (r1 - r0, n, 2)),
                                np.broadcast_to(ci[..., 1:2] - cjn[..., 3:4], (r1 - r0, n, 1)),
                                np.broadcast_to(cjn[..., 1:2] - ci[..., 3:4], (r1 - r0, n, 1))], -1)
            z = A[r0:r1, None, :] + B[None, :, :] + g @ Cg.T
            s[r0:r1] = (_gelu(z) @ w2[0] + b2[0])
        lg = (s - s.T) / 2
        p = 1 / (1 + np.exp(-np.clip(lg, -60, 60)))
        np.fill_diagonal(p, 0)
        gana = (p > 0.5).sum(1) + 1e-3 * p.sum(1)
        return [int(i) for i in np.argsort(-gana, kind="stable")]

    def _leer_con(self, lector, x, ls, sesgo):
        alto = lector.alto
        gs = [l["geo_o"][alto] for l in ls]
        recs = CC.recortes_pagina(x["orig"], gs, alto)
        lec = lector.leer(recs, self.presupuesto, sesgo)
        out = []
        for l, g, (t, c, confs, frames) in zip(ls, gs, lec):
            xs = []
            for f in frames:  # 4 px de recorte por fotograma (columnas de 16 px y 4 fotogramas por columna)
                col = int(min(g["W"] - 1, max(0, (f + 0.5) * 4)))
                xs.append(float(g["bx"][col]) / x["eo"])
            out.append((t, c, confs, xs))
        return out

    def leer_recortes(self, recortes, cual: str = "lec", sesgo: float = 0.0):
        """Renglones sueltos ya enderezados (PIL o uint8, papel claro, de cualquier alto) con el lector grande («lec») o el
        multilingüe («ml»): -> [(texto, conf, confs por carácter, ancho/alto del recorte)]. Para la lectura de prueba de la orientación (agente «orientacion»)."""
        lector = self.ml if cual == "ml" and self.ml is not None else self.lec
        alto, maxc = lector.alto, int(getattr(lector, "max_cols", 240))
        recs = []
        for r in recortes:
            a = np.asarray(r.convert("L") if hasattr(r, "convert") else r, np.float32)
            h, w = a.shape
            if h < 2 or w < 2:
                a = np.full((alto, 16), 255.0, np.float32)
            else:
                nw = max(16, min(16 * maxc, int(round(w * alto / h))))
                a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((nw, alto), Image.BILINEAR), np.float32)
                lo, hi = np.quantile(a, [0.01, 0.995])
                if hi - lo > 8:
                    a = np.clip((a - lo) * (255.0 / max(hi - lo, 1e-3)), 0, 255)
            w16 = int(math.ceil(a.shape[1] / 16)) * 16
            out = np.full((alto, w16), 255.0, np.float32)
            out[:, : a.shape[1]] = a
            recs.append(out)
        return [(t, c, confs, r.shape[1] / alto) for (t, c, confs, _), r in zip(lector.leer(recs, self.presupuesto, sesgo), recs)]

    def no_latina(self, textos) -> float:
        """Fracción de letras no latinas (ni griegas) de unas lecturas: el mismo criterio que el selector por escritura."""
        letras = [ch for t in textos for ch in t if unicodedata.category(ch)[0] == "L"]
        if len(letras) < 10:
            return 0.0
        return sum(escritura(ch) not in LATINAS for ch in letras) / len(letras)

    def leer_pagina(self, img):
        x = self._preparar(img)
        trab, estado = self._mapas(x)
        t0 = time.time()
        d = CC.decodificar_pagina(trab)
        self._t("cpu", t0)
        maqueta = "base"
        if self.densa is not None and len(d["lineas"]) >= self.densa.get("umbral", 200):
            # memoria (solo en CPU): la maquetación base sale mientras está la densa, y la densa sale al terminar la
            # página; en CUDA no, porque crear sesiones una y otra vez agota la memoria de la GPU
            if not self._en_gpu():
                self._s_mq = self._s_maq = None
                gc.collect()
            dz = self._sesiones_densa()
            trab, estado = self._mapas(x, densa=dz)
            t0 = time.time()
            d = CC.decodificar_pagina(trab)
            self._t("cpu", t0)
            maqueta = "densa"
        ls = d["lineas"]
        t0 = time.time()
        if maqueta == "densa" and len(ls) > 1:
            o = self._orden_jerarquico(estado, x, ls, dz["ord"], dz["op"])
        else:
            o = self._orden(estado, x, ls) if len(ls) > 1 else list(range(len(ls)))
        self._t("orden", t0)
        ls = [ls[k] for k in o]
        eo = x["eo"]
        for l in ls:
            l["geo_o"] = {}
            for alto, g in l["geo"].items():
                g = dict(g)
                if eo != 1:
                    for k in ("bx", "by"):
                        g[k] = g[k] * eo
                    for k in ("asc", "desc"):
                        g[k] = np.where(g[k] > -1e3, g[k] * eo, g[k])
                    g["v0"] *= eo
                    g["dv"] *= eo
                l["geo_o"][alto] = g
        sesgo = sesgo_blanco(self.arrojo)
        t0 = time.time()
        lect = self._leer_con(self.lec, x, ls, sesgo)
        quien = "lec"
        self._t("lector", t0)
        if self.ml is not None and ls:
            t0 = time.time()
            idx = sorted(range(len(ls)), key=lambda j: -(ls[j]["caja"][2] - ls[j]["caja"][0]))[:6]
            muestra = self._leer_con(self.ml, x, [ls[j] for j in idx], sesgo)
            letras = [ch for t, *_ in muestra for ch in t if unicodedata.category(ch)[0] == "L"]
            if len(letras) >= 10 and sum(escritura(ch) not in LATINAS for ch in letras) / len(letras) > self.umbral_ml:
                lect = self._leer_con(self.ml, x, ls, sesgo)
                quien = "ml"
            self._t("selector", t0)
        lineas = []
        for l, (t, c, confs, xs) in zip(ls, lect):
            t2, perm = a_logico_idx(t)
            lineas.append({"caja": [round(v, 1) for v in l["caja"]], "base": [[round(u, 1), round(v, 1)] for u, v in l["base"]],
                           "poligono": [[round(u, 1), round(v, 1)] for u, v in l["poligono"]],
                           "texto": t2, "conf": round(c, 4), "conf_linea": round(l["conf"], 4), "zona": l["zona"],
                           "zona_tipo": self.zonas[l["zona"]] if l["zona"] >= 0 else None,
                           "confs": [round(confs[i], 4) for i in perm], "xs": [round(xs[i], 1) for i in perm]})
        zonas = [dict(z, tipo=self.zonas[z["clase"]], caja=[round(v, 1) for v in z["caja"]], conf=round(z["conf"], 4)) for z in d["zonas"]]
        x.pop("_lin", None); x.pop("_est9", None)
        if maqueta == "densa" and not self._en_gpu():
            self._dz = None
            gc.collect()
        return {"lineas": lineas, "zonas": zonas, "tablas": d["tablas"], "lector": quien, "maqueta": maqueta,
                "texto": "\n".join(l["texto"] for l in lineas)}

    def leer(self, imagenes, lote=1):
        out = []
        for im in imagenes:
            if not isinstance(im, Image.Image):
                from .geometria_pagina import abrir

                im = abrir(im)  # con la orientación EXIF aplicada
            im.load()
            out.append(self.leer_pagina(im))
        return out
