"""Parte de CPU del modelo completo (sin torch ni scipy): mapas -> líneas, zonas y tablas; geometría y recorte
enderezado por línea base; corte XY. Es la misma lógica de la cadena de página del proyecto (pagina/cadena_cpu.py,
maqueta/modelo.py, detector.py, pagina/cadena.py), con los componentes conexos de OpenCV en lugar de scipy y el
grid_sample de la GPU hecho con cv2.remap.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

GROSOR = 0.45


# ---------------------------------------------------------------- componentes conexos
def _etiquetas(m: np.ndarray, conectividad: int):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=conectividad, ltype=cv2.CV_32S)
    return n - 1, lab, st  # st[i] = x, y, w, h, área (fila 0 = fondo)


def lineas_rapido(nuc: np.ndarray, lh: np.ndarray, umbral=0.5):
    """nuc [h, w] logits, lh [2, h, w] log1p alturas -> [(caja, conf, forma)] en celdas del mapa."""
    p = 1 / (1 + np.exp(-nuc.astype(np.float32)))
    n, lab, st = _etiquetas(p > umbral, 8)
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
        x0, y0, w, h = (int(v) for v in st[i, :4])
        x1, y1 = x0 + w, y0 + h
        if (y1 - y0) * (x1 - x0) < 3:
            continue
        a, b = inicio[i], inicio[i + 1]
        cols = kc[a:b]
        conf = float(psum[i] / area[i])
        vertical = (y1 - y0) > 2.5 * (x1 - x0)
        if vertical:
            msk = lab[y0:y1, x0:x1] == i
            m_ys, m_xs = np.nonzero(msk)
            d1 = float(np.expm1(lh[0][y0:y1, x0:x1][msk].astype(np.float32)).mean())
            d2 = float(np.expm1(lh[1][y0:y1, x0:x1][msk].astype(np.float32)).mean())
            grosor = area[i] / max(1, y1 - y0)
            ins = 0.15 * grosor / GROSOR
            cx_m = float(m_xs.mean() + x0)
            caja = [cx_m - d1, y0 - ins, cx_m + d2, y1 + ins]
            forma = {"base": [[cx_m, y0 - ins], [cx_m, y1 + ins]], "vertical": True,
                     "poligono": [[caja[0], caja[1]], [caja[2], caja[1]], [caja[2], caja[3]], [caja[0], caja[3]]]}
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
            forma = {"base": base, "poligono": pol, "vertical": False}
        out.append((caja, conf, forma))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1e-9)


def zonas_desde_mapas(zn: np.ndarray, zr: np.ndarray, umbral=0.4, nms=0.5, area_min=2):
    """zn [NZ,h,w] probabilidades; zr [NZ,4,h,w] log1p distancias -> [(clase, caja paso 4, confianza)]."""
    out = []
    for k in range(zn.shape[0]):
        n, lab, st = _etiquetas(zn[k] > umbral, 4)
        if n == 0:
            continue
        d = np.expm1(zr[k])
        for i in range(1, n + 1):
            x0, y0, w, h = (int(v) for v in st[i, :4])
            msk = lab[y0:y0 + h, x0:x0 + w] == i
            if msk.sum() < area_min:
                continue
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
            if all(iou(o[1], q[1]) < nms for q in keep):
                keep.append(o)
        final += keep
    return final


def tabla_desde_mapas(tb: np.ndarray, caja, umbral=0.5):
    x0, y0, x1, y1 = (int(round(v)) for v in caja)
    x0, y0 = max(0, x0), max(0, y0)
    sub = tb[:, y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
    if sub.size == 0:
        return [], []

    def cortes(perfil):
        out, i = [], 0
        while i < len(perfil):
            if perfil[i] > umbral:
                j = i
                while j < len(perfil) and perfil[j] > umbral:
                    j += 1
                out.append((i + j) / 2)
                i = j
            else:
                i += 1
        return out

    fy = cortes(sub[1].mean(1))
    fx = cortes(sub[2].mean(0))
    ys = [y0] + [y0 + c for c in fy] + [y1]
    xs = [x0] + [x0 + c for c in fx] + [x1]
    return list(zip(ys, ys[1:])), list(zip(xs, xs[1:]))


def corte_xy(cajas):
    """Corte XY recursivo original (filas y luego columnas): orden de respaldo y pista del orden aprendido."""
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

    def rec(idx, profundidad=0):
        if len(idx) <= 1 or profundidad > 30:
            return sorted(idx, key=lambda i: (cajas[i][1], cajas[i][0]))
        filas = cortar(idx, 0)
        if len(filas) > 1:
            return [j for g in filas for j in rec(g, profundidad + 1)]
        cols = cortar(idx, 1)
        if len(cols) > 1:
            return [j for g in cols for j in rec(g, profundidad + 1)]
        return sorted(idx, key=lambda i: (cajas[i][1], cajas[i][0]))

    import sys
    lim = sys.getrecursionlimit()
    sys.setrecursionlimit(max(lim, 10000))
    try:
        return rec(list(range(len(cajas))))
    finally:
        sys.setrecursionlimit(lim)


# ---------------------------------------------------------------- geometría del recorte enderezado
def _remuestrear(pts, paso=1.0):
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 2:
        return pts[:1].repeat(2, 0), s[-1]
    t = np.arange(0, s[-1], paso)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1), s[-1]


def geometria_recorte(base, poligono, alto, max_cols=240, margen_v=0.2, margen_h=0.5, suavizar=9):
    B = np.asarray(base, np.float32)
    P = np.asarray(poligono, np.float32)
    if len(B) < 2 or len(P) < 3:
        return None
    if B[-1, 0] < B[0, 0]:
        B = B[::-1]
    b, L = _remuestrear(B, 1.0)
    t = np.gradient(b, axis=0) if len(b) > 1 else np.array([[1.0, 0.0]] * len(b))
    if suavizar > 1 and len(t) > suavizar:
        k = np.ones(suavizar) / suavizar
        t = np.stack([np.convolve(t[:, 0], k, "same"), np.convolve(t[:, 1], k, "same")], 1)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-6)
    n = np.stack([t[:, 1], -t[:, 0]], 1)
    d2 = ((P[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    idx = np.argmin(d2, 1)
    dist = ((P - b[idx]) * n[idx]).sum(1)
    asc = max(4.0, float(dist.max()))
    desc = max(2.0, float(-dist.min()))
    h = asc + desc
    s_all = idx.astype(np.float32)

    def env(sel, vals, defecto):
        if sel.sum() == 0:
            return lambda s: np.full_like(s, defecto)
        ss, vv = s_all[sel], vals[sel]
        o = np.argsort(ss)
        ss, vv = ss[o], vv[o]
        return lambda s: np.interp(s, ss, vv)

    e_up = env(dist > 0, dist, asc)
    e_dn = env(dist < 0, -dist, desc)
    Hs = h * (1 + 2 * margen_v)
    k = alto / Hs
    Wout = int(round((L + 2 * margen_h * h) * k))
    Wout = max(16, min(Wout, max_cols * 16))
    kx = Wout / max(1e-6, (L + 2 * margen_h * h))
    s = (np.arange(Wout) + 0.5) / kx - margen_h * h
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
    a_up = e_up(si).astype(np.float32)
    a_dn = e_dn(si).astype(np.float32)
    fuera = (s < 0) | (s > L)
    a_up[fuera] = -1e4
    a_dn[fuera] = -1e4
    return {"bx": bx.astype(np.float32), "by": by.astype(np.float32), "nx": nx.astype(np.float32), "ny": ny.astype(np.float32),
            "asc": a_up, "desc": a_dn, "v0": float(asc + margen_v * h - 0.5 / k), "dv": float(1.0 / k), "W": Wout}


def decodificar_pagina(d):
    """d: {nuc, lh, esc_l, paso, umbral, zn, zr, tb, esc_m, altos, max_cols, nz_texto, tabla_idx} -> líneas, zonas, tablas."""
    f = d["paso"] / d["esc_l"]
    lineas = []
    for caja, conf, forma in lineas_rapido(d["nuc"], d["lh"], d["umbral"]):
        c = [max(0.0, v * f) for v in caja]
        if c[2] - c[0] <= 3 or c[3] - c[1] <= 3:
            continue
        lineas.append({"caja": c, "base": [[x * f, y * f] for x, y in forma["base"]],
                       "poligono": [[x * f, y * f] for x, y in forma["poligono"]], "conf": conf, "vertical": forma["vertical"]})
    zonas, tablas = [], []
    if d.get("zn") is not None:
        zn = d["zn"].astype(np.float32)
        zr = d["zr"].astype(np.float32).reshape(zn.shape[0], 4, zn.shape[1], zn.shape[2])
        fz = 4 / d["esc_m"]
        for k, b, c in zonas_desde_mapas(zn, zr):
            zonas.append({"clase": int(k), "caja": [v * fz for v in b], "conf": c})
            if k in d["tabla_idx"] and d.get("tb") is not None:
                filas, cols = tabla_desde_mapas(d["tb"].astype(np.float32), b)
                tablas.append({"caja": [v * fz for v in b], "filas": [[a * fz, bb * fz] for a, bb in filas],
                               "columnas": [[a * fz, bb * fz] for a, bb in cols]})
    zt = [z for z in zonas if z["clase"] in d["nz_texto"]]
    for l in lineas:
        c = l["caja"]
        area = max(1e-6, (c[2] - c[0]) * (c[3] - c[1]))
        mejor = None
        for z in zt:
            b = z["caja"]
            ix = max(0, min(c[2], b[2]) - max(c[0], b[0])) * max(0, min(c[3], b[3]) - max(c[1], b[1]))
            if ix / area > 0.5:
                a = (b[2] - b[0]) * (b[3] - b[1])
                if mejor is None or a < mejor[0]:
                    mejor = (a, z["clase"], b)
        l["zona"] = mejor[1] if mejor else -1
        l["zcaja"] = mejor[2] if mejor else c
        l["geo"] = {}
        for alto in d["altos"]:
            g = geometria_recorte(l["base"], l["poligono"], alto, d["max_cols"]) if not l["vertical"] else None
            if g is None:
                x0, y0, x1, y1 = c
                hh = max(2.0, y1 - y0)
                g = geometria_recorte([[x0, y1 - 0.2 * hh], [x1, y1 - 0.2 * hh]],
                                      [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], alto, d["max_cols"])
            l["geo"][alto] = g
    return {"lineas": lineas, "zonas": zonas, "tablas": tablas}


# ---------------------------------------------------------------- recorte enderezado (cv2.remap en lugar de grid_sample)
def _piramide(pag: np.ndarray, f: int) -> np.ndarray:
    """avg_pool2d(f, ceil_mode=True): media de bloques f×f, los del borde con los píxeles que caen dentro."""
    if f == 1:
        return pag.astype(np.float32)
    H, W = pag.shape
    Hn, Wn = -(-H // f), -(-W // f)
    a = np.zeros((Hn * f, Wn * f), np.float32)
    c = np.zeros_like(a)
    a[:H, :W] = pag
    c[:H, :W] = 1
    s = a.reshape(Hn, f, Wn, f).sum((1, 3))
    n = c.reshape(Hn, f, Wn, f).sum((1, 3))
    return s / np.maximum(n, 1)


def recortes_pagina(pag: np.ndarray, gs: list, alto: int, normalizar=True) -> list:
    """pag [H, W] uint8; gs: geometrías (ya a la escala de pag) -> lista de [alto, w16] float32 (0-255)."""
    if not gs:
        return []
    dv = float(np.median([g["dv"] for g in gs]))
    nivel = 0
    while dv / (2 ** (nivel + 1)) >= 1.0 and nivel < 4:
        nivel += 1
    f = 2 ** nivel
    src = _piramide(pag, f)
    out = []
    v_filas = np.arange(alto, dtype=np.float32)
    for g in gs:
        W = g["W"]
        v = g["v0"] - v_filas * g["dv"]  # [alto]
        X = g["bx"][None, :W] + v[:, None] * g["nx"][None, :W]
        Y = g["by"][None, :W] + v[:, None] * g["ny"][None, :W]
        dentro = (v[:, None] <= g["asc"][None, :W]) & (v[:, None] >= -g["desc"][None, :W])
        s = cv2.remap(src, (X / f - 0.5).astype(np.float32), (Y / f - 0.5).astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        sel = s[dentro]
        papel = float(np.quantile(sel, 0.9)) if sel.size else float(np.median(s))
        s = np.where(dentro, s, papel)
        if normalizar:
            med = float(np.median(s))
            if papel < med - 5 or papel < 100:
                s = 255.0 - s
            lo, hi = np.quantile(s, [0.01, 0.995])
            if hi - lo > 8:
                s = np.clip((s - lo) * (255.0 / max(hi - lo, 1e-3)), 0, 255)
            else:
                s = np.clip(s, 0, 255)
        w16 = int(math.ceil(W / 16)) * 16
        r = np.full((alto, w16), 255.0, np.float32)
        r[:, :W] = s
        out.append(r)
    return out


# ---------------------------------------------------------------- bloques del orden jerárquico (= maqueta/orden_jer.py)
MAX_BLOQUE = 600


def bloques_de(cajas, zid):
    """Lista de listas de índices de línea. zid[i] >= 0: zona; -1: se agrupa con vecinas sin zona de la misma columna.
    Los bloques de más de MAX_BLOQUE líneas se trocean por altura."""
    grupos, sueltas = {}, []
    for i, z in enumerate(zid):
        (grupos.setdefault(int(z), []) if z >= 0 else sueltas).append(i)
    out = list(grupos.values())
    if sueltas:
        cols = []  # [x0, x1, [índices]]
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
