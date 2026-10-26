"""Orden de lectura por grafo disperso (GCN): renglones = nodos, vecinos cercanos = aristas, paso de mensajes y dos
cabezas, como en «Text Reading Order in Uncontrolled Conditions by Sparse Graph Segmentation» (Wang, Fujii y Bissacco,
2023). Inferencia solo con numpy (pesos en orden_gcn.npz).

- Grafo: para cada renglón, sus K vecinos más cercanos por hueco entre cajas (en cuerpos) y el más cercano arriba,
  abajo, a la izquierda y a la derecha; aristas en los dos sentidos.
- Rasgos de nodo: caja, centro, tamaño (en página y en cuerpos), renglones en su fila y a su izquierda, puesto en su
  columna. Rasgos de arista (de i a j): desplazamientos, huecos, solapes y razones de tamaño.
- Capas: mensaje m_ij = relu(W [h_i, h_j, e_ij]); la arista se actualiza (e += m) y el nodo suma la media de los
  mensajes que le llegan (h += MLP [h, media]).
- Cabezas: «sucesor» en cada arista dirigida (j es el renglón que sigue a i) y «antes» por pares, factorizada
  (A·h_i + B·h_j + C·d_ij) y antisimétrica: s(i, j) = g(i, j) − g(j, i).
- Decodificación «agrupar y ordenar»: cadenas por sucesor (voraz, una entrada y una salida por renglón, sin ciclos) y
  orden de las cadenas con la media de s entre sus renglones y la búsqueda por inserciones.
"""
from __future__ import annotations

import os

import numpy as np

K = 8
N_NODO = 14
N_ARISTA = 12
AQUI = os.path.dirname(os.path.abspath(__file__))


def _geometria(b):
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = np.maximum(x1 - x0, 1.0)
    h = np.maximum(y1 - y0, 1.0)
    return x0, y0, x1, y1, w, h


def _par(b, mh, W, H, I, J):
    """Rasgos de arista de I hacia J (índices emparejados) -> [len(I), N_ARISTA]."""
    x0, y0, x1, y1, w, h = _geometria(b)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hov = np.maximum(0, np.minimum(x1[I], x1[J]) - np.maximum(x0[I], x0[J])) / np.minimum(w[I], w[J])
    vov = np.maximum(0, np.minimum(y1[I], y1[J]) - np.maximum(y0[I], y0[J])) / np.minimum(h[I], h[J])
    f = np.stack([(cx[J] - cx[I]) / mh / 20, (cy[J] - cy[I]) / mh / 20, (x0[J] - x0[I]) / mh / 20,
                  (x1[J] - x1[I]) / mh / 20, (y0[J] - y1[I]) / mh / 10, (x0[J] - x1[I]) / mh / 20, hov, vov,
                  np.log(h[J] / h[I]), np.log(w[J] / w[I]), (cx[J] - cx[I]) / W, (cy[J] - cy[I]) / H], 1)
    return np.clip(f, -3, 3).astype(np.float32)


def grafo(cajas, tam):
    """cajas [N, 4], tam (W, H) -> (X [N, N_NODO], src, dst, E [n_aristas, N_ARISTA], mh)."""
    b = np.asarray(cajas, np.float32).reshape(-1, 4)
    W, H = float(tam[0]), float(tam[1])
    n = len(b)
    x0, y0, x1, y1, w, h = _geometria(b)
    mh = float(np.median(h)) if n else 1.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hovm = np.maximum(0, np.minimum(x1[:, None], x1[None]) - np.maximum(x0[:, None], x0[None])) / np.minimum(w[:, None], w[None])
    vovm = np.maximum(0, np.minimum(y1[:, None], y1[None]) - np.maximum(y0[:, None], y0[None])) / np.minimum(h[:, None], h[None])
    fila = vovm > 0.5
    n_fila = fila.sum(1) - 1
    izq = (fila & (cx[None, :] < cx[:, None])).sum(1)
    rango = ((hovm > 0.3) & (cy[None, :] < cy[:, None])).sum(1)
    X = np.stack([x0 / W, y0 / H, x1 / W, y1 / H, cx / W, cy / H, w / W, np.log(h / mh), np.log(w / mh) / 5,
                  np.full(n, mh / H * 20), np.full(n, W / H), np.minimum(n_fila, 12) / 4, np.minimum(izq, 12) / 4,
                  np.minimum(rango, 60) / 20], 1).astype(np.float32)
    # vecinos: hueco entre cajas en cuerpos (más un poco de distancia entre centros para desempatar)
    dx = np.maximum(0, np.maximum(x0[None, :] - x1[:, None], x0[:, None] - x1[None, :])) / mh
    dy = np.maximum(0, np.maximum(y0[None, :] - y1[:, None], y0[:, None] - y1[None, :])) / mh
    d = dx + dy + 0.02 * (np.abs(cx[None] - cx[:, None]) + np.abs(cy[None] - cy[:, None])) / mh
    np.fill_diagonal(d, np.inf)
    pares = set()
    k = min(K, n - 1)
    if k > 0:
        vec = np.argpartition(d, k - 1, axis=1)[:, :k]
        for i in range(n):
            for j in vec[i]:
                pares.add((min(i, j), max(i, j)))
        arriba = (cy[None, :] < cy[:, None]) & (hovm > 0)
        abajo = (cy[None, :] > cy[:, None]) & (hovm > 0)
        izqm = (cx[None, :] < cx[:, None]) & (vovm > 0.3)
        der = (cx[None, :] > cx[:, None]) & (vovm > 0.3)
        for m in (arriba, abajo, izqm, der):
            dd = np.where(m, d, np.inf)
            j = dd.argmin(1)
            for i in np.where(np.isfinite(dd[np.arange(n), j]))[0]:
                pares.add((min(i, j[i]), max(i, j[i])))
    if pares:
        p = np.array(sorted(pares), np.int64)
        src = np.concatenate([p[:, 0], p[:, 1]])
        dst = np.concatenate([p[:, 1], p[:, 0]])
    else:
        src = dst = np.zeros(0, np.int64)
    E = _par(b, mh, W, H, src, dst)
    return X, src, dst, E, mh


def pares_todos(cajas, tam, mh):
    """Rasgos de todos los pares [N, N, N_ARISTA] (de i hacia j)."""
    b = np.asarray(cajas, np.float32).reshape(-1, 4)
    n = len(b)
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return _par(b, mh, float(tam[0]), float(tam[1]), I.ravel(), J.ravel()).reshape(n, n, N_ARISTA)


# ---------------------------------------------------------------- red (numpy)
_M = None


PESOS = ("orden_gcn.npz", "orden_gcn_giro.npz")  # el de siempre y el entrenado con inclinación residual (GCN_GIRO=2)
PESO_CONJUNTO = (2.0, 1.0)  # dos tercios del de siempre y un tercio del de los giros (a partes iguales, Grotowski 6,13)


def modelo():
    """Pesos empaquetados: la pareja de PESOS (se promedian sus logits; véase `red_conjunto`), el primero solo si falta
    el segundo, o None si no hay ninguno."""
    global _M
    if _M is None:
        ms = [cargar(os.path.join(AQUI, r)) for r in PESOS if os.path.exists(os.path.join(AQUI, r))]
        _M = (ms[0] if len(ms) == 1 else ms) if ms and os.path.exists(os.path.join(AQUI, PESOS[0])) else False
    return _M or None


def cargar(ruta=None):
    d = np.load(ruta or os.path.join(AQUI, "orden_gcn.npz"))
    return {k: (d[k].astype(np.float32) if d[k].dtype.kind == "f" else d[k]) for k in d.files}  # guardados en fp16


def _relu(x):
    return np.maximum(x, 0)


def red(m, X, src, dst, E, P):
    """-> (logits de sucesor por arista dirigida [n_aristas], s antisimétrica [N, N])."""
    n = len(X)
    h = _relu(X @ m["n0w"] + m["n0b"]) @ m["n1w"] + m["n1b"]
    e = E @ m["e0w"] + m["e0b"]
    grado = np.maximum(np.bincount(dst, minlength=n), 1).astype(np.float32)[:, None]
    for l in range(int(m["capas"])):
        msg = _relu(np.concatenate([h[src], h[dst], e], 1) @ m[f"m{l}w"] + m[f"m{l}b"])
        e = e + msg
        agg = np.zeros_like(h)
        np.add.at(agg, dst, msg)
        agg /= grado
        h = h + _relu(np.concatenate([h, agg], 1) @ m[f"u{l}w"] + m[f"u{l}b"]) @ m[f"v{l}w"] + m[f"v{l}b"]
    suc = (_relu(np.concatenate([h[src], h[dst], e], 1) @ m["s0w"] + m["s0b"]) @ m["s1w"] + m["s1b"])[:, 0]
    a = h @ m["pa"]
    bb = h @ m["pb"]
    g = (_relu(a[:, None, :] + bb[None, :, :] + P @ m["pc"] + m["pbias"]) @ m["pw"])[..., 0]
    return suc, g - g.T


def red_conjunto(m, X, src, dst, E, P):
    """`red` con un modelo o con una lista de modelos: media ponderada (PESO_CONJUNTO) de los logits de sucesor y de los
    pares. La pareja aguanta la inclinación residual (Máscara, p. 63, girada 2,9° sin enderezar: WER 26,4 -> 15,3 %) sin
    perder en la validación recta (97,20 · 60,2 % -> 97,26 · 60,8 %); el de los giros solo bajaba a 97,11 · 59,3 %."""
    if isinstance(m, dict):
        return red(m, X, src, dst, E, P)
    outs = [red(x, X, src, dst, E, P) for x in m]
    w = np.array(PESO_CONJUNTO[:len(outs)], np.float32)
    w /= w.sum()
    return sum(wi * o[0] for wi, o in zip(w, outs)), sum(wi * o[1] for wi, o in zip(w, outs))


# ---------------------------------------------------------------- decodificación
def decodificar(a, barridos=4):
    """Orden a partir de logits por pares a[i, j] (> 0: i antes que j): arranque por suma de log-probabilidades y
    búsqueda local por inserciones (la de orden.py)."""
    n = a.shape[0]
    if n <= 1:
        return list(range(n))
    lp = -np.logaddexp(0.0, -a)
    orden = list(np.argsort(-(lp.sum(1) - lp.sum(0)), kind="stable"))
    for _ in range(barridos):
        mejora = False
        for x in list(orden):
            resto = [y for y in orden if y != x]
            r = np.array(resto)
            pref = np.concatenate([[0.0], np.cumsum(lp[r, x])])
            suf = np.concatenate([np.cumsum(lp[x, r][::-1])[::-1], [0.0]])
            p = int(np.argmax(pref + suf))
            nuevo = resto[:p] + [x] + resto[p:]
            if nuevo != orden:
                orden, mejora = nuevo, True
        if not mejora:
            break
    return [int(i) for i in orden]


def cadenas(suc, src, dst, n, umbral=0.0):
    """Cadenas de renglones por el sucesor: aristas de más a menos probables, una salida y una entrada por renglón,
    sin ciclos. -> lista de cadenas (listas de índices)."""
    sig = -np.ones(n, np.int64)
    ant = -np.ones(n, np.int64)
    raiz = list(range(n))

    def buscar(x):
        while raiz[x] != x:
            raiz[x] = raiz[raiz[x]]
            x = raiz[x]
        return x

    for k in np.argsort(-suc, kind="stable"):
        if suc[k] <= umbral:
            break
        i, j = int(src[k]), int(dst[k])
        if sig[i] >= 0 or ant[j] >= 0:
            continue
        ri, rj = buscar(i), buscar(j)
        if ri == rj:
            continue
        sig[i], ant[j] = j, i
        raiz[ri] = rj
    out = []
    for i in range(n):
        if ant[i] < 0:
            c = [i]
            while sig[c[-1]] >= 0:
                c.append(int(sig[c[-1]]))
            out.append(c)
    return out


def ordenar_de(suc, s, src, dst, modo="cadenas"):
    n = s.shape[0]
    if modo == "pares" or n <= 2:
        return decodificar(s)
    cs = cadenas(suc, src, dst, n)
    if len(cs) == 1:
        return cs[0]
    M = np.array([[s[np.ix_(a, b)].mean() if a is not b else 0.0 for b in cs] for a in cs], np.float64)
    return [i for k in decodificar(M) for i in cs[k]]


def ordenar(m, cajas, tam, textos=None, modo="cadenas"):
    n = len(cajas)
    if n <= 1:
        return list(range(n))
    X, src, dst, E, mh = grafo(cajas, tam)
    P = pares_todos(cajas, tam, mh)
    suc, s = red_conjunto(m, X, src, dst, E, P)
    return ordenar_de(suc, s, src, dst, modo)
