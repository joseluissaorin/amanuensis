"""articulos_red (de maqueta/articulos_red): agrupa en ARTÍCULOS los bloques de una página de prensa con la cabeza aprendida «mismo
artículo» por pares y una partición global por agrupamiento de correlación (B-cubed 88,05 en la mitad B de val de GRESEL; reglas v1: 79,8).
Opcional (`--articulos red`): piezas de ~960 MB (catálogo «articulos-red» o AMANUENSIS_RUTA_ARTICULOS_RED), dependencia `tokenizers`
(si falta, el motor avisa y usa las reglas), 6,6 s/pág. en CPU del Mac (visión 3,9 s). API: Articulos(carpeta, hilos).pagina(pag_dict, imagen_PIL)
-> [{"bloques": [...], "lineas": [...], "titulo": str, "confianza": p}].
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
MARCAS = re.compile(r"(contin[uú]a|viene de|pasa a|sigue en|termina en)", re.I)


def _sesion(ruta, hilos):
    so = ort.SessionOptions()
    so.intra_op_num_threads = hilos
    so.inter_op_num_threads = 1
    return ort.InferenceSession(ruta, so, providers=["CPUExecutionProvider"])


def _mayus(t):
    l = [c for c in t if c.isalpha()]
    return sum(c.isupper() for c in l) / max(1, len(l))


# ---------------------------------------------------------------- partición (como particion.py)
def _particion(n, pares, probs, reinicios=12, falta=-0.5):
    if n == 0:
        return np.zeros(0, int)
    W = np.full((n, n), falta, np.float64)
    p = np.clip(np.asarray(probs, np.float64), 1e-4, 1 - 1e-4)
    for (i, j), v in zip(pares, np.log(p / (1 - p))):
        W[i, j] = W[j, i] = v
    np.fill_diagonal(W, 0.0)

    def pivote(orden):
        et = -np.ones(n, int)
        g = 0
        for i in orden:
            if et[i] >= 0:
                continue
            libres = (et < 0) & (W[i] > 0)
            libres[i] = True
            et[libres] = g
            g += 1
        return et

    def local(et, max_it=100):
        et = et.copy()
        idx = np.arange(n)
        for _ in range(max_it):
            mejoro = False
            for i in range(n):
                otros = idx != i
                gs = np.unique(et[otros])
                v = {g: W[i, otros & (et == g)].sum() for g in gs}
                solo = not (otros & (et == et[i])).any()
                actual = 0.0 if solo else v.get(et[i], 0.0)
                g_m = max(v, key=v.get) if v else None
                if g_m is not None and v[g_m] > max(actual, 0.0) + 1e-9:
                    et[i] = g_m
                    mejoro = True
                elif actual < -1e-9:
                    et[i] = et.max() + 1
                    mejoro = True
            gs = list(np.unique(et))
            for x in range(len(gs)):
                for y in range(x + 1, len(gs)):
                    ma, mb = et == gs[x], et == gs[y]
                    if ma.any() and mb.any() and W[np.ix_(ma, mb)].sum() > 1e-9:
                        et[mb] = gs[x]
                        mejoro = True
            if not mejoro:
                break
        _, et = np.unique(et, return_inverse=True)
        return et

    rnd = np.random.default_rng(0)
    mejor, mv = None, -np.inf
    for o in [np.arange(n)] + [rnd.permutation(n) for _ in range(reinicios - 1)]:
        et = local(pivote(o))
        m = et[:, None] == et[None, :]
        v = float((W * m).sum() / 2)
        if v > mv:
            mejor, mv = et, v
    return mejor


class _CodArtCoreML:
    """cod_art en Core ML (producto/coreml/cod_art_cm.py): una función por cubo de alto, página rellena de blanco; carpeta `coreml_cod_art`."""

    def __init__(self, carpeta):
        from .completo_coreml import _Funciones, disponible

        if not disponible():
            raise RuntimeError("Core ML no disponible")
        self.cfg = json.load(open(os.path.join(carpeta, "config.json")))
        self.altos, self.esc = sorted(self.cfg["altos"]), float(self.cfg.get("escala_estados", 64.0))
        self.f = _Funciones(os.path.join(carpeta, self.cfg["paquete"]), max_vivas=1)

    def estados(self, x):
        """x [1, h, 1024, 3] en 0..1 -> [1, (h/16)*64, 768] o None si la página es más alta que el mayor cubo."""
        h = x.shape[1]
        Hc = next((a for a in self.altos if a >= h), None)
        if Hc is None or x.shape[2] != self.cfg["ancho"]:
            return None
        g = np.ones((1, Hc, x.shape[2], 3), np.float32)
        g[:, :h] = x
        e = self.f(f"h{Hc}").predict({"pixeles": g})["estados"]
        est = e[:, : (h // 16) * (x.shape[2] // 16)].astype(np.float32) * self.esc
        self.f.vaciar()  # se usa una vez por página: soltar sus pesos mientras leen el lector y la maqueta
        return est


class Articulos:
    def __init__(self, carpeta: str, hilos: int = 4):
        self.c = carpeta
        self.meta = json.load(open(os.path.join(carpeta, "articulos.json")))
        self._cod, self._hilos = None, hilos
        self.cm = None
        cmd = os.path.join(carpeta, "coreml_cod_art")
        if os.path.isdir(cmd) and os.environ.get("AMANUENSIS_COREML", "1") != "0":
            try:
                self.cm = _CodArtCoreML(cmd)
            except Exception:  # noqa: BLE001 (sin Core ML: ONNX)
                self.cm = None
        self.txt = _sesion(os.path.join(carpeta, "texto", "texto_q8.onnx"), hilos)
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(carpeta, "texto", "tokenizer.json"))
        self.lm = _sesion(os.path.join(carpeta, "lm_fp32.onnx"), hilos)
        self.lmc = json.load(open(os.path.join(carpeta, "lm.json")))
        self.lm_idx = {ch: i + self.lmc["especiales"] for i, ch in enumerate(self.lmc["chars"])}
        self.lm_pliegue = str.maketrans(self.lmc["pliegue"])
        self.cab = _sesion(os.path.join(carpeta, "cabeza.onnx"), hilos)

    @property
    def cod(self):
        if self._cod is None:
            self._cod = _sesion(os.path.join(self.c, "cod_art_q8.onnx"), self._hilos)
        return self._cod

    # ---- rasgos (como maqueta/articulos_red/rasgos.py)
    @staticmethod
    def _bloques(pag):
        ls = pag["lineas"]
        out = []
        for k, b in enumerate(pag.get("bloques", [])):
            ii = [i for i in b["lineas"] if 0 <= i < len(ls)]
            if ii:
                out.append((k, [ls[i] for i in ii]))
        return out

    @staticmethod
    def _geo(bl, W, H):
        hs = np.array([l["caja"][3] - l["caja"][1] for _, b in bl for l in b], float)
        h = float(np.median(hs)) if len(hs) else 1.0
        ws = np.array([l["caja"][2] - l["caja"][0] for _, b in bl for l in b], float)
        col = float(np.median(ws[hs <= 1.2 * h])) if (hs <= 1.2 * h).any() else float(np.median(ws)) if len(ws) else W / 5
        G, cajas, textos = [], [], []
        for _, b in bl:
            c = [min(l["caja"][0] for l in b), min(l["caja"][1] for l in b), max(l["caja"][2] for l in b), max(l["caja"][3] for l in b)]
            hb = float(np.median([l["caja"][3] - l["caja"][1] for l in b]))
            ts = [(l.get("texto") or "").strip() for l in b]
            t = " ".join(x for x in ts if x)
            ult = ts[-1] if ts else ""
            G.append([c[0] / W, c[1] / H, c[2] / W, c[3] / H, (c[2] - c[0]) / max(1, col), (c[3] - c[1]) / H, hb / h,
                      min(len(ts), 60) / 60, min(len(t), 3000) / 3000, _mayus(t), float(ult.endswith(("-", "¬"))),
                      float(ult.endswith((".", "»", '"', ")"))), float(bool(MARCAS.search(t))), (c[0] / max(1, col)) / 10])
            cajas.append(c)
            textos.append(t)
        return np.array(G, np.float32), np.array(cajas, np.float32), textos, h, col

    def _vis(self, img, cajas, W, H):
        anc = self.meta["ancho_vis"]
        s = anc / img.width
        w, h = (int(img.width * s) // 16) * 16, (int(img.height * s) // 16) * 16
        x = np.asarray(img.convert("RGB").resize((w, h)), np.float32)[None] / 255.0
        est = self.cm.estados(x) if self.cm is not None else None
        est = (est if est is not None else self.cod.run(None, {"pixeles": x})[0])[0]  # [N, 768], fila a fila
        gw = w // 16
        xs, ys = np.arange(est.shape[0]) % gw, np.arange(est.shape[0]) // gw
        out = []
        for c in cajas:
            x0, y0, x1, y1 = (c[0] * w / W) / 16, (c[1] * h / H) / 16, (c[2] * w / W) / 16, (c[3] * h / H) / 16
            m = (xs >= int(x0)) & (xs <= int(x1)) & (ys >= int(y0)) & (ys <= int(y1))
            out.append(est[m].mean(0) if m.any() else est.mean(0))
        return np.stack(out).astype(np.float16).astype(np.float32)

    def _texto(self, textos, lote=16):
        pre, mx = self.meta["prefijo_texto"], self.meta["max_texto"]
        xs = [pre + (t[:mx] or "·") for t in textos]
        out = []
        for i in range(0, len(xs), lote):
            enc = self.tok.encode_batch(xs[i:i + lote])
            T = max(len(e.ids) for e in enc)
            ids = np.zeros((len(enc), T), np.int64)
            msk = np.zeros((len(enc), T), np.int64)
            pad = self.tok.token_to_id("<pad>") or 0
            ids[:] = pad
            for k, e in enumerate(enc):
                ids[k, : len(e.ids)] = e.ids
                msk[k, : len(e.ids)] = 1
            hid = self.txt.run(None, {"input_ids": ids, "attention_mask": msk})[0]
            m = msk[..., None].astype(np.float32)
            e = (hid * m).sum(1) / m.sum(1)
            out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float16).astype(np.float32)

    def _lm_cod(self, t):
        return [self.lm_idx.get(c, self.lmc["DESC"]) for c in t.translate(self.lm_pliegue)]

    def _lm_pares(self, pares, lote=128):
        L = self.lmc
        seqs = []
        for ant, act in pares:
            a = self._lm_cod(ant)[-L["MAX_ANT"]:]
            b = self._lm_cod(act)[: L["MAX_ACT"]]
            ids = [L["INI"]] + a + [L["SEP"]] + b + [L["FIN"]]
            peso = [0] * (len(a) + 1) + [1] * (len(b) + 1)
            seqs.append((ids, peso))
        out = np.zeros(len(seqs), np.float32)
        orden = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
        for i in range(0, len(orden), lote):
            idx = orden[i:i + lote]
            T = max(len(seqs[j][0]) for j in idx)
            X = np.full((len(idx), T), L["PAD"], np.int64)
            M = np.zeros((len(idx), T), np.float32)
            for k, j in enumerate(idx):
                ids, peso = seqs[j]
                X[k, : len(ids)] = ids
                M[k, : len(peso)] = peso
            out[idx] = self.lm.run(None, {"X": X, "M": M})[0]
        return out

    def _entre(self, g, a, b, W, H):
        gh, gw = g.shape
        sx, sy = gw / W, gh / H
        out = [0.0, 0.0, 0.0]
        if a[3] <= b[1]:
            x0, x1 = int(max(a[0], b[0]) * sx), int(min(a[2], b[2]) * sx)
            y0, y1 = int(a[3] * sy), int(b[1] * sy)
            if x1 - x0 > 4 and y1 - y0 > 0:
                f = g[y0:y1, x0:x1] < 110
                out[0] = float(f.mean(1).max()) if f.size else 0.0
                out[2] = (y1 - y0) / gh
        if a[2] <= b[0] or b[2] <= a[0]:
            l, r = (a, b) if a[2] <= b[0] else (b, a)
            x0, x1 = int(l[2] * sx), int(r[0] * sx)
            y0, y1 = int(max(a[1], b[1]) * sy), int(min(a[3], b[3]) * sy)
            if x1 - x0 > 0 and y1 - y0 > 4:
                f = g[y0:y1, x0:x1] < 110
                out[1] = float(f.mean(0).max()) if f.size else 0.0
        return out

    def _pares(self, cajas, G, textos, g, W, H, h):
        n, k = len(cajas), self.meta["k_vecinos"]
        idx = []
        for i in range(n):
            for j in range(i + 1, n):
                mismo = min(cajas[i][2], cajas[j][2]) - max(cajas[i][0], cajas[j][0]) > 0.5 * min(cajas[i][2] - cajas[i][0], cajas[j][2] - cajas[j][0])
                if j - i <= k or mismo or G[j][12] > 0 or G[i][12] > 0:
                    idx.append((i, j))
        F = []
        for i, j in idx:
            a, b = cajas[i], cajas[j]
            sol = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(1.0, min(a[2] - a[0], b[2] - b[0]))
            F.append([(b[0] - a[0]) / W, (b[1] - a[3]) / H, (b[0] - a[2]) / W, sol, (j - i) / max(1, n), float(sol > 0.5),
                      float(sol > 0.5 and 0 <= b[1] - a[3] < 3 * h), *self._entre(g, a, b, W, H)])
        lmc = np.zeros((len(idx), 2), np.float32)
        if idx:
            ini = [textos[j][:80] for _, j in idx]
            con = self._lm_pares([(textos[i][-120:], x) for (i, _), x in zip(idx, ini)])
            sin = self._lm_pares([("", x) for x in ini])
            lmc[:, 0] = (con - sin) / np.maximum(1, np.array([len(x) for x in ini]))
        lmc[:, 1] = [float(textos[i].rstrip().endswith(("-", "¬"))) for i, _ in idx]
        return np.array(idx, np.int64).reshape(-1, 2), np.concatenate([np.array(F, np.float32).reshape(-1, 10), lmc], 1)

    def pagina(self, pag: dict, img: Image.Image) -> list[dict]:
        bl = self._bloques(pag)
        if len(bl) < 2:
            return [{"bloques": [k for k, _ in bl], "titulo": "", "confianza": 1.0}] if bl else []
        W, H = pag["ancho"], pag["alto"]
        G, cajas, textos, h, col = self._geo(bl, W, H)
        g = np.asarray(img.convert("L").resize((min(2048, img.width), int(img.height * min(2048, img.width) / img.width))))
        I, FP = self._pares(cajas, G, textos, g, W, H, h)
        vis = self._vis(img, cajas, W, H)
        txt = self._texto(textos)
        lg = self.cab.run(None, {"geo": G, "vis": vis, "txt": txt, "pares": I, "fpar": FP})[0] + self.meta["sesgo"]
        pr = 1 / (1 + np.exp(-lg))
        et = _particion(len(bl), I, pr, falta=self.meta["falta"])
        grupos = {}
        for (kb, b), e in zip(bl, et):
            grupos.setdefault(int(e), []).append(kb)
        conf = {}
        for (i, j), q in zip(I, pr):
            if et[i] == et[j]:
                conf.setdefault(int(et[i]), []).append(float(q))
        out = []
        for e, ks in sorted(grupos.items(), key=lambda x: x[1][0]):
            t = next((textos[i] for i, (kb, _) in enumerate(bl) if kb == ks[0]), "")
            lineas = [i for kb in ks for i in pag["bloques"][kb]["lineas"] if 0 <= i < len(pag["lineas"])]
            out.append({"bloques": ks, "lineas": lineas, "titulo": t[:120], "confianza": round(float(np.mean(conf.get(e, [1.0]))), 3)})
        return out
