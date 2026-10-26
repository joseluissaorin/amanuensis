# Copia de maqueta/minima/inferir.py (agente «maqueta-minima»); si cambia allí, se vuelve a copiar.
"""Inferencia de la maquetación mínima SIN PyTorch ni scipy (numpy + onnxruntime + opencv): es la referencia que siguen el
paquete (producto/amanuensis) y la demo web (web/prueba/src/maqueta_minima.ts).

    from inferir import MaquetaMinima
    mm = MaquetaMinima("resultados/mm_onnx")          # maqueta.onnx, orden.onnx, config.json
    r = mm.analizar(pil_img)                          # orientación, renglones, zonas y orden
    r = {"giro": grados antihorarios aplicados (PIL), "imagen": la página derecha (PIL, gris), "ancho", "alto",
         "lineas": [{"caja", "base" (polilínea), "poligono", "asc", "desc" (alto sobre y bajo la base, px), "tipo",
                     "zona" (índice en «zonas» o -1), "conf"}]   # EN ORDEN DE LECTURA
         "zonas": [{"tipo", "caja", "conf"}], "orientacion": [p0, p90, p180, p270]}
    recorte = mm.recorte(r["imagen"], linea)          # renglón ENDEREZADO por su línea base (PIL, gris)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from PIL import Image

ZONAS = ["texto", "parrafo", "margen", "titulo", "figura", "tabla", "numero", "capitular", "cabecera", "pie", "nota", "formula",
         "lista", "leyenda", "musica", "sello"]
LINEAS = ["normal", "titulo", "interlineal", "capitular", "musica"]
TEXTO_IDX = {ZONAS.index(z) for z in ("texto", "parrafo", "margen", "titulo", "cabecera", "pie", "nota", "lista", "leyenda", "numero")}
NZ, NL = len(ZONAS), len(LINEAS)
GROSOR = 0.45
MAX_BLOQUE = 400


# ---------------------------------------------------------------- decodificación de renglones (= pagina/cadena_cpu.lineas_rapido)
def _etiquetas(m: np.ndarray):
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return n - 1, lab, st


def lineas_desde_mapas(nuc: np.ndarray, lh: np.ndarray, lt: np.ndarray | None = None, umbral=0.5):
    """nuc [h, w] logits, lh [2, h, w], lt [NL, h, w] -> [(caja, conf, forma, tipo)] en celdas del mapa."""
    p = 1 / (1 + np.exp(-nuc.astype(np.float32)))
    n, lab, st = _etiquetas(p > umbral)
    if n == 0:
        return []
    H, W = p.shape
    ys, xs = np.nonzero(lab)
    L = lab[ys, xs]
    up_px = np.expm1(lh[0][ys, xs].astype(np.float32))
    dn_px = np.expm1(lh[1][ys, xs].astype(np.float32))
    pv = p[ys, xs]
    area = np.bincount(L, minlength=n + 1).astype(np.float64)
    psum = np.bincount(L, weights=pv, minlength=n + 1)
    tipo_s = None
    if lt is not None:
        tipo_s = np.stack([np.bincount(L, weights=lt[k][ys, xs].astype(np.float64), minlength=n + 1) for k in range(lt.shape[0])], 1)
    clave = L.astype(np.int64) * W + xs
    uk, inv = np.unique(clave, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    sy = np.bincount(inv, weights=ys) / cnt
    su = np.bincount(inv, weights=up_px) / cnt
    sd = np.bincount(inv, weights=dn_px) / cnt
    kl = uk // W
    kc = uk % W
    inicio = np.searchsorted(kl, np.arange(n + 2))
    out = []
    for i in range(1, n + 1):
        x0, y0, w_, h_ = (int(v) for v in st[i, :4])
        x1, y1 = x0 + w_, y0 + h_
        if w_ * h_ < 3:
            continue
        a, b = inicio[i], inicio[i + 1]
        cols = kc[a:b]
        conf = float(psum[i] / area[i])
        tipo = int(np.argmax(tipo_s[i])) if tipo_s is not None else 0
        vertical = h_ > 2.5 * w_
        if vertical:
            msk = lab[y0:y1, x0:x1] == i
            m_ys, m_xs = np.nonzero(msk)
            d1 = float(np.expm1(lh[0][y0:y1, x0:x1][msk].astype(np.float32)).mean())
            d2 = float(np.expm1(lh[1][y0:y1, x0:x1][msk].astype(np.float32)).mean())
            grosor = area[i] / max(1, h_)
            ins = 0.15 * grosor / GROSOR
            cx = float(m_xs.mean() + x0)
            caja = [cx - d1, y0 - ins, cx + d2, y1 + ins]
            forma = {"base": [[cx, y0 - ins], [cx, y1 + ins]], "vertical": True,
                     "poligono": [[caja[0], caja[1]], [caja[2], caja[1]], [caja[2], caja[3]], [caja[0], caja[3]]],
                     "asc": [d1, d1], "desc": [d2, d2]}
        else:
            yv, uv, dv = sy[a:b], su[a:b], sd[a:b]
            paso = max(1, len(cols) // 12)
            idx = list(range(0, len(cols), paso)) + [len(cols) - 1]
            mid = [[float(cols[k]), float(yv[k])] for k in idx]
            up_k = [float(uv[k]) for k in idx]
            dn_k = [float(dv[k]) for k in idx]
            ins = 0.15 * float(np.median(np.array(up_k) + np.array(dn_k)))
            mid[0][0] = float(cols[0]) - ins
            mid[-1][0] = float(cols[-1]) + ins
            arriba = [[x, y - u] for (x, y), u in zip(mid, up_k)]
            abajo = [[x, y + d] for (x, y), d in zip(mid, dn_k)]
            base = [[x, y + 0.6 * d] for (x, y), d in zip(mid, dn_k)]
            pol = arriba + abajo[::-1]
            caja = [min(q[0] for q in pol), min(q[1] for q in pol), max(q[0] for q in pol), max(q[1] for q in pol)]
            forma = {"base": base, "poligono": pol, "vertical": False,
                     "asc": [u + 0.6 * d for u, d in zip(up_k, dn_k)], "desc": [0.4 * d for d in dn_k]}
        out.append((caja, conf, forma, tipo))
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def zonas_desde_mapas(zn: np.ndarray, zr: np.ndarray, umbral=0.4, nms=0.5, area_min=2):
    """= maqueta/modelo.zonas_desde_mapas con opencv. zn [NZ,h,w] probabilidades; zr [NZ,4,h,w]."""
    out = []
    for k in range(zn.shape[0]):
        m = zn[k] > umbral
        if not m.any():
            continue
        n, lab, st = _etiquetas(m)
        d = np.expm1(zr[k].astype(np.float32))
        for i in range(1, n + 1):
            if st[i, 4] < area_min:
                continue
            x0, y0, w_, h_ = (int(v) for v in st[i, :4])
            msk = lab[y0:y0 + h_, x0:x0 + w_] == i
            ys, xs = np.nonzero(msk)
            ys = ys + y0
            xs = xs + x0
            pr = zn[k, ys, xs]
            ww = pr / pr.sum()
            caja = [float(((xs - d[0, ys, xs]) * ww).sum()), float(((ys - d[1, ys, xs]) * ww).sum()),
                    float(((xs + d[2, ys, xs]) * ww).sum()), float(((ys + d[3, ys, xs]) * ww).sum())]
            out.append((k, caja, float(pr.mean())))
    final = []
    for k in sorted({o[0] for o in out}):
        cs = sorted([o for o in out if o[0] == k], key=lambda o: -o[2])
        keep = []
        for o in cs:
            if all(_iou(o[1], q[1]) < nms for q in keep):
                keep.append(o)
        final += keep
    return final


def unir_fragmentos(lin, hueco=0.6):
    """Une los trozos de un mismo renglón partidos en un hueco (rayas, espacios de justificación): misma banda vertical,
    altura parecida, línea base continua y hueco horizontal < `hueco` alturas. lin: dicts con caja_l, base_l, poligono_l,
    asc_l, desc_l (px del lienzo); se devuelve la lista nueva."""
    def alto(l):
        return max(1.0, l["caja_l"][3] - l["caja_l"][1])

    lin = sorted(lin, key=lambda l: l["caja_l"][0])
    hecho = True
    while hecho:
        hecho = False
        for i, a in enumerate(lin):
            if a.get("vertical"):
                continue
            for j in range(i + 1, len(lin)):
                b = lin[j]
                if b.get("vertical"):
                    continue
                ca, cb = a["caja_l"], b["caja_l"]
                ha, hb = alto(a), alto(b)
                h = min(ha, hb)
                gap = cb[0] - ca[2]
                if gap > hueco * h:
                    continue
                if gap < -0.3 * h or max(ha, hb) > 1.6 * h:
                    continue
                solape = min(ca[3], cb[3]) - max(ca[1], cb[1])
                if solape < 0.6 * h or abs(a["base_l"][-1][1] - b["base_l"][0][1]) > 0.35 * h:
                    continue
                na, nb = len(a["base_l"]), len(b["base_l"])
                pa, pb = a["poligono_l"], b["poligono_l"]
                arriba = pa[:na] + pb[:nb]
                abajo = pa[na:][::-1] + pb[nb:][::-1]
                pol = arriba + abajo[::-1]
                u = {**a, "caja_l": [min(ca[0], cb[0]), min(ca[1], cb[1]), max(ca[2], cb[2]), max(ca[3], cb[3])],
                     "base_l": a["base_l"] + b["base_l"], "poligono_l": pol, "asc_l": a["asc_l"] + b["asc_l"],
                     "desc_l": a["desc_l"] + b["desc_l"], "conf": (a["conf"] + b["conf"]) / 2,
                     "tipo": a["tipo"] if (ca[2] - ca[0]) >= (cb[2] - cb[0]) else b["tipo"]}
                lin = lin[:i] + [u] + lin[i + 1:j] + lin[j + 1:]
                hecho = True
                break
            if hecho:
                break
    return lin


# ---------------------------------------------------------------- orden jerárquico
def corte_xy(cajas):
    """Corte XY recursivo (= detector.corte_xy_original; las cajas en px del LIENZO de la red, como al entrenar)."""
    if len(cajas) <= 1:
        return list(range(len(cajas)))

    def cortar(idx, eje):
        a0, a1 = (1, 3) if eje == 0 else (0, 2)
        iv = sorted(idx, key=lambda i: cajas[i][a0])
        grupos, actual, fin = [], [iv[0]], cajas[iv[0]][a1]
        for i in iv[1:]:
            if cajas[i][a0] > fin - 0.1 * (cajas[i][a1] - cajas[i][a0]) and (eje == 0 or cajas[i][a0] > fin + 4):
                grupos.append(actual)
                actual = [i]
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


def rango_xy(cajas):
    n = len(cajas)
    if n <= 1:
        return [0.0] * n
    r = [0.0] * n
    for k, i in enumerate(corte_xy(cajas)):
        r[i] = k / (n - 1)
    return r


def bloques_de(cajas, zid):
    """= maqueta/orden_jer.bloques_de (zonas; renglones sin zona agrupados por columna; trozos de MAX_BLOQUE)."""
    grupos, sueltas = {}, []
    for i, z in enumerate(zid):
        (grupos.setdefault(int(z), []) if z >= 0 else sueltas).append(i)
    out = list(grupos.values())
    if sueltas:
        cols = []
        for i in sorted(sueltas, key=lambda i: cajas[i][1]):
            c = cajas[i]
            h = max(1.0, c[3] - c[1])
            for col in cols:
                ult = cajas[col[2][-1]]
                solape = min(c[2], col[1]) - max(c[0], col[0])
                if solape > 0.5 * min(c[2] - c[0], col[1] - col[0]) and c[1] - ult[3] < 2.5 * h:
                    col[2].append(i)
                    col[0], col[1] = min(col[0], c[0]), max(col[1], c[2])
                    break
            else:
                cols.append([c[0], c[2], [i]])
        out += [col[2] for col in cols]
    fin = []
    for b in out:
        if len(b) <= MAX_BLOQUE:
            fin.append(b)
        else:
            b = sorted(b, key=lambda i: cajas[i][1])
            fin += [b[k:k + MAX_BLOQUE] for k in range(0, len(b), MAX_BLOQUE)]
    return fin


def media_cajas(f8: np.ndarray, cajas: np.ndarray):
    """f8 [C, h, w], cajas [n, 4] en celdas -> [n, C] (= maqueta/modelo.media_cajas, imagen integral)."""
    C, H, W = f8.shape
    ii = np.zeros((C, H + 1, W + 1), np.float64)
    ii[:, 1:, 1:] = f8.astype(np.float64).cumsum(1).cumsum(2)
    x0 = np.clip(np.floor(cajas[:, 0]), 0, W - 1).astype(int)
    y0 = np.clip(np.floor(cajas[:, 1]), 0, H - 1).astype(int)
    x1 = np.maximum(np.clip(np.ceil(cajas[:, 2]), 1, W).astype(int), x0 + 1)
    y1 = np.maximum(np.clip(np.ceil(cajas[:, 3]), 1, H).astype(int), y0 + 1)
    s = ii[:, y1, x1] - ii[:, y0, x1] - ii[:, y1, x0] + ii[:, y0, x0]
    return (s / ((x1 - x0) * (y1 - y0))).T.astype(np.float32)


def rasgos(f8, cajas, cajas_b, zona, tipo, W, H):
    """Ítems -> [n, NF] (= maqueta/minima/entrenar.rasgos_items con datos.geo_items)."""
    c = np.asarray(cajas, np.float32)
    pool = media_cajas(f8, c / 8)
    cn = c / np.array([W, H, W, H], np.float32)
    b = np.asarray(cajas_b, np.float32) / np.array([W, H, W, H], np.float32)
    geo = np.concatenate([cn, cn[:, 2:] - cn[:, :2], (cn[:, :2] + cn[:, 2:]) / 2, b, cn[:, :2] - b[:, :2], b[:, 2:] - cn[:, 2:],
                          np.asarray(rango_xy(c.tolist()), np.float32)[:, None]], 1)
    oz = np.eye(NZ + 1, dtype=np.float32)[np.asarray(zona, int)]
    ol = np.eye(NL, dtype=np.float32)[np.asarray(tipo, int)]
    return np.concatenate([pool, geo, oz, ol], 1).astype(np.float32), cn


def ordenar_por_pares(lg: np.ndarray):
    p = 1 / (1 + np.exp(-lg))
    np.fill_diagonal(p, 0)
    gana = (p > 0.5).sum(1) + 1e-3 * p.sum(1)
    return list(np.argsort(-gana, kind="stable"))


# ---------------------------------------------------------------- recorte enderezado (= pagina/cadena_cpu.geometria_recorte + muestreo)
def _remuestrear(pts, paso=1.0):
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 2:
        return pts[:1].repeat(2, 0), s[-1]
    t = np.arange(0, s[-1], paso)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1), s[-1]


def rejilla(base, asc, desc, margen_v=0.25, margen_h=0.6, alto=None, suavizar=9):
    """Rejilla de muestreo (mapx, mapy) [alto, ancho] del renglón enderezado: columnas a lo largo de la línea base
    (a 1 px de arco por columna si alto es None), filas a lo largo de la normal suavizada; alto del renglón = asc + desc
    medianos, con márgenes como imagen.recorte_de_caja (0,25 alturas arriba y abajo, 0,6 a los lados)."""
    B = np.asarray(base, np.float32)
    if len(B) < 2:
        return None
    if B[-1, 0] < B[0, 0]:
        B = B[::-1]
    b, Lb = _remuestrear(B, 1.0)
    t = np.gradient(b, axis=0) if len(b) > 1 else np.array([[1.0, 0.0]] * len(b))
    if suavizar > 1 and len(t) > suavizar:
        k = np.ones(suavizar) / suavizar
        t = np.stack([np.convolve(t[:, 0], k, "same"), np.convolve(t[:, 1], k, "same")], 1)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-6)
    n = np.stack([t[:, 1], -t[:, 0]], 1)  # normal hacia ARRIBA de la página (y crece hacia abajo)
    a = max(3.0, float(np.median(asc)))
    d = max(1.0, float(np.median(desc)))
    h = a + d
    Hs = h * (1 + 2 * margen_v)
    k = 1.0 if alto is None else alto / Hs
    Wout = max(8, int(round((Lb + 2 * margen_h * h) * k)))
    Hout = max(8, int(round(Hs * k)))
    s = (np.arange(Wout) + 0.5) / k - margen_h * h
    nb = len(b)
    si = np.clip(s, 0, nb - 1)
    i0 = np.floor(si).astype(int)
    i1 = np.minimum(i0 + 1, nb - 1)
    fr = si - i0
    bx = b[i0, 0] * (1 - fr) + b[i1, 0] * fr
    by = b[i0, 1] * (1 - fr) + b[i1, 1] * fr
    nx = n[i0, 0] * (1 - fr) + n[i1, 0] * fr
    ny = n[i0, 1] * (1 - fr) + n[i1, 1] * fr
    t0, t1 = t[0], t[-1]
    ant = s < 0
    bx[ant] = b[0, 0] + s[ant] * t0[0]
    by[ant] = b[0, 1] + s[ant] * t0[1]
    des = s > nb - 1
    bx[des] = b[-1, 0] + (s[des] - (nb - 1)) * t1[0]
    by[des] = b[-1, 1] + (s[des] - (nb - 1)) * t1[1]
    v = (a + margen_v * h) - (np.arange(Hout) + 0.5) / k  # distancia a la base (positiva hacia arriba)
    mapx = bx[None, :] + v[:, None] * nx[None, :]
    mapy = by[None, :] + v[:, None] * ny[None, :]
    return mapx.astype(np.float32), mapy.astype(np.float32)


def recortar(gris: np.ndarray, linea: dict, **kw) -> np.ndarray:
    """gris [H, W] uint8 -> renglón enderezado [alto, ancho] uint8 (bilineal, borde replicado)."""
    import cv2
    if linea.get("vertical") or len(linea.get("base", [])) < 2:
        x0, y0, x1, y1 = linea["caja"]
        h = max(1.0, y1 - y0)
        mv, mh = kw.get("margen_v", 0.25), kw.get("margen_h", 0.6)
        xa, ya, xb, yb = int(x0 - mh * h), int(y0 - mv * h), int(x1 + mh * h), int(y1 + mv * h)
        out = np.full((max(1, yb - ya), max(1, xb - xa)), 255, np.uint8)
        H, W = gris.shape
        sa, sb, ta, tb = max(0, xa), min(W, xb), max(0, ya), min(H, yb)
        if sb > sa and tb > ta:
            out[ta - ya: tb - ya, sa - xa: sb - xa] = gris[ta:tb, sa:sb]
        return out
    r = rejilla(linea["base"], linea["asc"], linea["desc"], **kw)
    return cv2.remap(gris, r[0], r[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _orden_ligero():
    """El orden ligero del paquete (MLP de 27 KB sobre cajas), si está a mano."""
    try:
        from amanuensis import orden as O
    except Exception:
        try:
            from . import orden as O  # dentro del paquete
        except Exception:
            return None
    m = O.modelo()
    return None if m is None else (m, O.ordenar)


# ---------------------------------------------------------------- modelo
class MaquetaMinima:
    def __init__(self, carpeta: str, hilos: int = 0, proveedores=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if hilos:
            so.intra_op_num_threads = hilos
        prov = proveedores or ["CPUExecutionProvider"]
        self.cfg = json.load(open(os.path.join(carpeta, "config.json")))
        self.red = ort.InferenceSession(os.path.join(carpeta, self.cfg.get("maqueta", "maqueta.onnx")), so, providers=prov)
        self.ord = ort.InferenceSession(os.path.join(carpeta, self.cfg.get("orden", "orden.onnx")), so, providers=prov)
        self.lado = int(self.cfg.get("lado", 1024))
        self.umbral = float(self.cfg.get("umbral", 0.5))
        self.umbral_giro = float(self.cfg.get("umbral_giro", 0.8))
        self.umbral_derecha = float(self.cfg.get("umbral_derecha", 0.5))
        self.umbral_zona = float(os.environ.get("MM_UMBRAL_ZONA", self.cfg.get("umbral_zona", 0.4)))

    def mapas(self, g: np.ndarray, lado=None):
        """g [H, W] uint8 -> salidas de la red y escala (lienzo / original)."""
        lado = lado or self.lado
        H0, W0 = g.shape
        esc = lado / max(W0, H0)
        w, h = max(32, round(W0 * esc)), max(32, round(H0 * esc))
        Wc, Hc = int(math.ceil(w / 32)) * 32, int(math.ceil(h / 32)) * 32
        x = np.ones((1, 1, Hc, Wc), np.float32)
        x[0, 0, :h, :w] = np.asarray(Image.fromarray(g).resize((w, h), Image.BILINEAR), np.float32) / 255.0
        nombres = [o.name for o in self.red.get_outputs()]
        sal = dict(zip(nombres, self.red.run(None, {"gris": x})))
        return sal, esc, (Wc, Hc), (w, h)

    def orientacion(self, sal):
        z = sal["ori"][0].astype(np.float64)
        e = np.exp(z - z.max())
        return e / e.sum()

    def analizar(self, img: Image.Image, orientar=True, lado=None, puntuar=None):
        """puntuar(r) -> verosimilitud de la lectura de un análisis (p. ej. orientacion.verosimil sobre los renglones más
        largos leídos con el lector): si el clasificador duda (p(0°) < umbral_derecha), se prueban 0° y los dos giros
        más probables y gana la lectura más verosímil (cambiar de sentido tiene que ganar por 0,15)."""
        g = np.asarray(img.convert("L"))
        sal, esc, (Wc, Hc), _ = self.mapas(g, lado)
        po = self.orientacion(sal)
        if orientar and po[0] < self.umbral_derecha:
            if puntuar is not None:
                cands = [0] + [int(k) + 1 for k in np.argsort(-po[1:])[:2]]
                mejor = None
                for k in cands:
                    gk = np.ascontiguousarray(np.rot90(g, -k))
                    r = self.analizar(Image.fromarray(gk), orientar=False, lado=lado)
                    pt = puntuar(r) - (0.15 if k else 0.0)
                    if mejor is None or pt > mejor[0]:
                        mejor = (pt, k, r)
                r = mejor[2]
                r["giro"] = (360 - 90 * mejor[1]) % 360
                r["orientacion"] = [float(v) for v in po]
                return r
            k = int(np.argmax(po[1:])) + 1
            g = np.ascontiguousarray(np.rot90(g, -k))  # deshace el giro antihorario de k cuartos
            giro = (360 - 90 * k) % 360
            sal, esc, (Wc, Hc), _ = self.mapas(g, lado)
        else:
            giro = 0
        H0, W0 = g.shape
        # renglones (paso 2), en px del lienzo y luego del original
        ls = lineas_desde_mapas(sal["nuc"][0, 0], sal["lh"][0], sal["lt"][0], self.umbral)
        f = 2.0
        lin = []
        for caja, conf, forma, tipo in ls:
            c = [v * f for v in caja]
            if c[2] - c[0] <= 3 * esc or c[3] - c[1] <= 2 * esc:
                continue
            lin.append({"caja_l": c, "base_l": [[x * f, y * f] for x, y in forma["base"]],
                        "poligono_l": [[x * f, y * f] for x, y in forma["poligono"]],
                        "asc_l": [v * f for v in forma["asc"]], "desc_l": [v * f for v in forma["desc"]],
                        "tipo": tipo, "conf": conf, "vertical": forma["vertical"]})
        if self.cfg.get("unir", True):
            lin = unir_fragmentos(lin)
        # zonas (paso 4)
        zn = 1 / (1 + np.exp(-sal["zn"][0].astype(np.float32)))
        zr = sal["zr"][0].reshape(NZ, 4, zn.shape[1], zn.shape[2])
        zonas = [{"k": k_, "caja_l": [v * 4 for v in c], "conf": s} for k_, c, s in zonas_desde_mapas(zn, zr)]
        # zona de texto de cada renglón: la más pequeña que cubre más de la mitad de su caja
        for l in lin:
            c = l["caja_l"]
            area = max(1e-6, (c[2] - c[0]) * (c[3] - c[1]))
            mejor = None
            for j, z in enumerate(zonas):
                if z["k"] not in TEXTO_IDX or z["conf"] < self.umbral_zona:
                    continue
                b = z["caja_l"]
                ix = max(0, min(c[2], b[2]) - max(c[0], b[0])) * max(0, min(c[3], b[3]) - max(c[1], b[1]))
                if ix / area > 0.5:
                    a_ = (b[2] - b[0]) * (b[3] - b[1])
                    if mejor is None or a_ < mejor[0]:
                        mejor = (a_, j)
            l["zona"] = mejor[1] if mejor else -1
        modo = os.environ.get("MM_ORDEN", self.cfg.get("orden_modo", "jerarquico"))
        if modo == "ligero" and _orden_ligero() is not None and 1 < len(lin) <= 400:
            m_, ordenar_ = _orden_ligero()
            orden = ordenar_(m_, [l["caja_l"] for l in lin], (Wc, Hc))
        else:
            zz = [l["zona"] for l in lin]
            if modo == "columnas":  # sin zonas: bloques por columna
                for l in lin:
                    l["zona"] = -1
            orden = self.ordenar(sal["f8"][0], lin, zonas, Wc, Hc)
            for l, z in zip(lin, zz):
                l["zona"] = z
        lin = [lin[i] for i in orden]
        e = 1 / esc
        lineas = [{"caja": [v * e for v in l["caja_l"]], "base": [[x * e, y * e] for x, y in l["base_l"]],
                   "poligono": [[x * e, y * e] for x, y in l["poligono_l"]], "asc": [v * e for v in l["asc_l"]],
                   "desc": [v * e for v in l["desc_l"]], "tipo": LINEAS[l["tipo"]], "zona": l["zona"], "conf": l["conf"],
                   "vertical": l["vertical"]} for l in lin]
        zs = [{"tipo": ZONAS[z["k"]], "caja": [v * e for v in z["caja_l"]], "conf": z["conf"]} for z in zonas]
        return {"giro": giro, "imagen": Image.fromarray(g), "ancho": W0, "alto": H0, "lineas": lineas, "zonas": zs,
                "orientacion": [float(v) for v in po]}

    def _orden_items(self, x, nc):
        if len(x) <= 1:
            return list(range(len(x)))
        lg = self.ord.run(None, {"x": x, "nc": nc})[0]
        return ordenar_por_pares(lg)

    def ordenar(self, f8, lin, zonas, Wc, Hc):
        n = len(lin)
        if n <= 1:
            return list(range(n))
        cajas = [l["caja_l"] for l in lin]
        zid = [l["zona"] for l in lin]
        bl = bloques_de(cajas, zid)
        cb = [[min(cajas[i][0] for i in b), min(cajas[i][1] for i in b), max(cajas[i][2] for i in b), max(cajas[i][3] for i in b)] for b in bl]
        zcl = [0 if l["zona"] < 0 else zonas[l["zona"]]["k"] + 1 for l in lin]
        if len(bl) > 1:
            x, nc = rasgos(f8, cb, [[0, 0, Wc, Hc]] * len(bl), [zcl[b[0]] for b in bl], [0] * len(bl), Wc, Hc)
            ob = self._orden_items(x, nc)
        else:
            ob = [0]
        out = []
        for k in ob:
            b = bl[k]
            if len(b) == 1:
                out += b
                continue
            x, nc = rasgos(f8, [cajas[i] for i in b], [cb[k]] * len(b), [zcl[i] for i in b], [lin[i]["tipo"] for i in b], Wc, Hc)
            out += [b[j] for j in self._orden_items(x, nc)]
        return out

    def recorte(self, img, linea, **kw) -> Image.Image:
        g = img if isinstance(img, np.ndarray) else np.asarray(img.convert("L"))
        return Image.fromarray(recortar(g, linea, **kw))
