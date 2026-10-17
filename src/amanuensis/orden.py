"""Orden de lectura ligero para la vía CPU: un perceptrón diminuto por pares de líneas sobre geometría con
contexto de página, más una decodificación por inserciones. Inferencia solo con numpy (pesos en .npz).

Rasgos de una pareja (i, j): posición y tamaño de cada línea; desplazamientos relativos; solapes vertical y
horizontal; cuántas líneas quedan entre las dos en la misma columna de cada una; si hay entre ellas una línea
que abarca a ambas (título o filete que separa franjas); número de líneas en la fila de cada una. La puntuación es
antisimétrica: s(i, j) = f(i, j) - f(j, i).
"""
from __future__ import annotations

import os

import numpy as np

N_RASGOS = 28
AQUI = os.path.dirname(os.path.abspath(__file__))


def rasgos(cajas, tam):
    """cajas [N, 4], tam (W, H) -> tensor [N, N, N_RASGOS] float32 (rasgos de i antes que j)."""
    b = np.asarray(cajas, np.float32).reshape(-1, 4)
    W, H = float(tam[0]), float(tam[1])
    n = len(b)
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = np.maximum(x1 - x0, 1.0); h = np.maximum(y1 - y0, 1.0)
    mh = float(np.median(h)) if n else 1.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # solapes por pares
    hov = np.maximum(0, np.minimum(x1[:, None], x1[None]) - np.maximum(x0[:, None], x0[None])) / np.minimum(w[:, None], w[None])
    vov = np.maximum(0, np.minimum(y1[:, None], y1[None]) - np.maximum(y0[:, None], y0[None])) / np.minimum(h[:, None], h[None])
    fila = (vov > 0.5).sum(1) - 1  # líneas en la misma fila
    izq = ((vov > 0.5) & (cx[None, :] < cx[:, None])).sum(1)  # a su izquierda en la fila
    # entre i y j (en vertical): k con cy entre las dos
    ent = (cy[None, :, None] > np.minimum(cy[:, None], cy[None])[:, None, :] + 1e-3) & \
          (cy[None, :, None] < np.maximum(cy[:, None], cy[None])[:, None, :] - 1e-3)  # [i, k, j]
    ent_i = (ent & (hov[:, :, None] > 0.3)).sum(1)  # en la columna de i
    ent_j = (ent & (hov.T[None, :, :] > 0.3)).sum(1)  # en la columna de j
    abarca = (x0[None, :, None] <= np.minimum(x0[:, None], x0[None])[:, None, :] + mh) & \
             (x1[None, :, None] >= np.maximum(x1[:, None], x1[None])[:, None, :] - mh)
    sep = (ent & abarca).any(1)
    # orden de y dentro de la «columna» (líneas que solapan en horizontal)
    rango = np.array([((hov[i] > 0.3) & (cy < cy[i])).sum() for i in range(n)], np.float32)
    uno = np.stack([x0 / W, y0 / H, x1 / W, y1 / H, w / W, np.log(h / mh), fila / 4.0, izq / 4.0, rango / 20.0,
                    np.full(n, mh / H * 20)], 1).astype(np.float32)  # 10
    I = np.broadcast_to(uno[:, None, :], (n, n, 10)); J = np.broadcast_to(uno[None, :, :], (n, n, 10))
    par = np.stack([(cx[None] - cx[:, None]) / W, (cy[None] - cy[:, None]) / H,
                    np.clip((y0[None] - y1[:, None]) / mh, -20, 20) / 10, np.clip((x0[None] - x1[:, None]) / mh, -40, 40) / 20,
                    vov, hov, (ent_i + ent_j) / 20.0, sep.astype(np.float32)], 2).astype(np.float32)  # 8
    return np.concatenate([I, J, par], 2)


_M = None


def modelo():
    """Pesos empaquetados (orden_ligero.npz, unos KB) o None si no están."""
    global _M
    if _M is None:
        ruta = os.path.join(AQUI, "orden_ligero.npz")
        _M = cargar(ruta) if os.path.exists(ruta) else False
    return _M or None


def cargar(ruta=None):
    d = np.load(ruta or os.path.join(AQUI, "orden_ligero.npz"))
    return {k: d[k] for k in d.files}


def puntuar(m, F):
    """MLP numpy: F [..., N_RASGOS] -> logit."""
    x = (F - m["media"]) / m["desv"]
    for k in range(int(m["capas"])):
        x = x @ m[f"w{k}"] + m[f"b{k}"]
        if k < int(m["capas"]) - 1:
            x = np.maximum(x, 0)
    return x[..., 0]


def decodificar(a, barridos=4):
    """Orden a partir de logits por pares a[i, j] (> 0: i antes que j): arranque por suma de log-probabilidades
    y búsqueda local por inserciones (sacar cada línea y volver a ponerla donde más sube la verosimilitud)."""
    n = a.shape[0]
    if n <= 1:
        return list(range(n))
    lp = -np.logaddexp(0.0, -a)  # log P(i antes que j)
    orden = list(np.argsort(-(lp.sum(1) - lp.sum(0))))
    for _ in range(barridos):
        mejora = False
        for x in list(orden):
            resto = [y for y in orden if y != x]
            r = np.array(resto)
            antes = lp[r, x]   # r antes que x
            despues = lp[x, r]  # x antes que r
            # puntuación de poner x en la posición p: sum_{q<p} antes[q] + sum_{q>=p} despues[q]
            pref = np.concatenate([[0.0], np.cumsum(antes)])
            suf = np.concatenate([np.cumsum(despues[::-1])[::-1], [0.0]])
            p = int(np.argmax(pref + suf))
            nuevo = resto[:p] + [x] + resto[p:]
            if nuevo != orden:
                orden, mejora = nuevo, True
        if not mejora:
            break
    return [int(i) for i in orden]


def ordenar(m, cajas, tam, textos=None):
    n = len(cajas)
    if n <= 1:
        return list(range(n))
    F = rasgos(cajas, tam)
    f = puntuar(m, F)
    return decodificar(f - f.T)
