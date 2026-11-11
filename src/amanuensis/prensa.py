"""Prensa a varias columnas (modo «prensa»): renglones apretados, canales estrechos y cabeceras que abarcan columnas.

Se aplica después del modelo completo, sobre los renglones ya leídos (`--prensa si`, o `auto`: páginas con tres o más
columnas de texto). Nació con el corpus de prensa histórica de GRESEL (UNED, noviembre de 2026: La Vanguardia de
Manila, Fémina, Hispanidad, Semana…), donde la página entera salía con las columnas intercaladas.

1. **Renglones dobles.** Con interlineado muy apretado, el detector junta a veces dos o tres renglones en una caja y el
   lector solo lee uno. Una caja mucho más alta que sus vecinas de columna se parte por el perfil de tinta y cada trozo
   se vuelve a leer con el lector grande.
2. **Renglones que cruzan el canal.** Cuando dos columnas tienen los renglones a la misma altura, a veces salen unidos.
   El canal se deduce de los vecinos (muchos acaban a un lado y empiezan al otro, casi ninguno lo cruza) y el renglón se
   corta por el carácter más cercano, con la x de cada carácter que da el lector.
3. **Orden de lectura** con las reglas de Breuel (2003): a antes que b si (1) se solapan en x y a está encima, o (2) a
   está entero a la izquierda de b y ningún renglón entre los dos en altura se solapa con ambos (una cabecera a varias
   columnas separa lo de arriba de lo de abajo). Orden topológico.
4. **Bloques**: renglones seguidos de la misma columna, sin hueco grande y del mismo cuerpo de letra, pasan a ser una
   zona propia (así la salida en párrafos, TEI, PAGE o Markdown sigue la columna y no la zona de la maqueta).
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

from .salidas import Linea, Pagina

NO_TEXTO = {"GraphicZone", "Picture", "StampZone", "SealZone", "MusicZone", "DigitizationArtefactZone", "DamageZone",
            "Formula"}
TABLA = {"Table", "TableZone"}


def _alto_cuerpo(cajas: np.ndarray, textos) -> float:
    hs = [c[3] - c[1] for c, t in zip(cajas, textos) if len(t.strip()) >= 5]
    return float(np.median(hs)) if hs else 20.0


# ---------------------------------------------------------------- detección
def columnas(lineas, ancho: float) -> int:
    """Número de columnas de texto corrido: grupos de renglones del cuerpo con el mismo comienzo (±1 alto) y anchos
    parecidos; cuentan los grupos con 8 renglones o más."""
    ls = [l for l in lineas if len(l.texto.strip()) >= 8]
    if len(ls) < 30:
        return 1
    C = np.array([l.caja for l in ls], float)
    h = _alto_cuerpo(C, [l.texto for l in ls])
    w = C[:, 2] - C[:, 0]
    wm = float(np.median(w))
    if wm > 0.6 * ancho:
        return 1
    cuerpo = (np.abs(w - wm) < 0.35 * wm) & ((C[:, 3] - C[:, 1]) < 1.6 * h)
    x0 = np.sort(C[cuerpo, 0])
    grupos, act = [], [x0[0]] if len(x0) else []
    for x in x0[1:]:
        if x - act[-1] <= max(4.0, 0.8 * h):
            act.append(x)
        else:
            grupos.append(act)
            act = [x]
    if act:
        grupos.append(act)
    return sum(len(g) >= 8 for g in grupos)


def es_prensa(p: Pagina) -> bool:
    return columnas(p.lineas, p.ancho) >= int(os.environ.get("AMANUENSIS_PRENSA_COLUMNAS", "3"))


# ---------------------------------------------------------------- 1. renglones dobles
def _cortes_perfil(g: np.ndarray, h_ref: float, pico: float = 0.35, valle: float = 0.55, sep: float = 0.42) -> list[tuple[int, int]]:
    """Recorte gris (papel claro) de una caja alta -> franjas [y0, y1) de cada renglón: picos del perfil de tinta (la
    banda de las minúsculas de cada renglón) separados por valles claros; [] si no hay al menos dos."""
    if g.size == 0:
        return []
    a = g.astype(np.float32)
    lo, hi = np.quantile(a, [0.02, 0.98])
    if hi - lo < 20:
        return []
    p = (np.clip((hi - a) / (hi - lo), 0, 1) > 0.45).astype(np.float32).mean(1)
    k = max(1, int(round(h_ref * 0.08)))
    p = np.convolve(p, np.ones(k) / k, mode="same")
    m = float(p.max())
    if m <= 0:
        return []
    H = len(p)
    picos = [y for y in range(1, H - 1) if p[y] >= p[y - 1] and p[y] > p[y + 1] and p[y] >= pico * m]
    sel = []
    for y in sorted(picos, key=lambda y: -p[y]):
        if all(abs(y - s) >= sep * h_ref for s in sel):
            sel.append(y)
    sel.sort()
    if len(sel) < 2:
        return []
    cortes = []
    for a_, b_ in zip(sel, sel[1:]):
        v = a_ + int(np.argmin(p[a_:b_ + 1]))
        if p[v] >= valle * min(p[a_], p[b_]):
            return []
        cortes.append(v)
    out, ini = [], 0
    for c in cortes + [H]:
        out.append((ini, c))
        ini = c
    return out


def _leer(completo, grises: list[np.ndarray]):
    """Recortes grises -> [(texto, conf, confs, xs_rel)] con el lector grande; xs_rel en fracción del ancho del recorte."""
    from .completo import sesgo_blanco

    lector = completo.lec
    alto, maxc = lector.alto, int(getattr(lector, "max_cols", 240))
    recs, nws = [], []
    for a in grises:
        a = a.astype(np.float32)
        h, w = a.shape
        nw = max(16, min(16 * maxc, int(round(w * alto / max(h, 1)))))
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((nw, alto), Image.BILINEAR), np.float32)
        lo, hi = np.quantile(a, [0.01, 0.995])
        if hi - lo > 8:
            a = np.clip((a - lo) * (255.0 / max(hi - lo, 1e-3)), 0, 255)
        w16 = int(math.ceil(nw / 16)) * 16
        out = np.full((alto, w16), 255.0, np.float32)
        out[:, :nw] = a
        recs.append(out)
        nws.append(nw)
    res = lector.leer(recs, completo.presupuesto, sesgo_blanco(completo.arrojo))
    return [(t, c, confs, [min(1.0, (f * 4 + 2) / nw) for f in fr]) for (t, c, confs, fr), nw in zip(res, nws)]


def _parecido(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    a, b = "".join(a.split()).lower(), "".join(b.split()).lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def partir_dobles(lineas: list, img: Image.Image, completo, factor: float = 1.2) -> tuple[list, int]:
    """Renglones con la caja más alta que un renglón sencillo y con dos o más bandas de tinta -> varios renglones
    releídos. Las franjas del borde que son un resto del renglón vecino se descartan si ese renglón ya está leído; si
    no (renglón perdido por el detector), se recuperan."""
    if len(lineas) < 10 or completo is None:
        return lineas, 0
    C = np.array([l.caja for l in lineas], float)
    textos = [l.texto for l in lineas]
    h = _alto_cuerpo(C, textos)
    alto = C[:, 3] - C[:, 1]
    cy = (C[:, 1] + C[:, 3]) / 2
    con_texto = np.array([len(t.strip()) >= 5 for t in textos])
    g = np.asarray(img.convert("L"))
    H, W = g.shape
    trabajos = []  # (índice, [(caja, es_borde)])
    for i, l in enumerate(lineas):
        x0, y0, x1, y1 = l.caja
        if x1 - x0 < 3 * h:
            continue
        sx = np.minimum(C[:, 2], x1) - np.maximum(C[:, 0], x0)
        vec = (sx > 0.5 * np.minimum(C[:, 2] - C[:, 0], x1 - x0)) & (np.abs(cy - cy[i]) < 15 * h) & (np.arange(len(C)) != i) & con_texto
        ref = min(h, float(np.quantile(alto[vec], 0.3))) if vec.sum() >= 3 else h
        if alto[i] < factor * ref:
            continue
        ya, yb = max(0, int(y0)), min(H, int(math.ceil(y1)))
        xa, xb = max(0, int(x0)), min(W, int(math.ceil(x1)))
        franjas = _cortes_perfil(g[ya:yb, xa:xb], ref)
        if len(franjas) < 2:
            continue
        med = float(np.median([b - a for a, b in franjas]))
        cajas = []
        for k, (a, b) in enumerate(franjas):
            borde = (k == 0 or k == len(franjas) - 1) and (b - a) < 0.6 * med
            if borde:  # se extiende hacia fuera hasta el alto de una banda
                if k == 0:
                    a = b - med
                else:
                    b = a + med
            m = 0.12 * (b - a)
            cajas.append(([x0, max(0.0, ya + a - m), x1, min(float(H), ya + b + m)], borde))
        trabajos.append((i, cajas))
    if not trabajos:
        return lineas, 0
    grises = [g[int(c[1]):int(math.ceil(c[3])), max(0, int(c[0])):min(W, int(math.ceil(c[2])))] for _, cs in trabajos for c, _ in cs]
    lect = _leer(completo, grises)
    nuevas, k, n_partidas = {}, 0, 0
    for i, cs in trabajos:
        nucleo, bordes = [], []
        for c, borde in cs:
            t, conf, confs, xr = lect[k]
            k += 1
            t = t.strip() if t else ""
            if sum(ch.isalnum() for ch in t) < 2:
                continue
            xs = [c[0] + v * (c[2] - c[0]) for v in xr]
            tz = Linea([round(v, 1) for v in c], t, round(float(conf), 4), [round(v, 1) for v in xs][: len(t)],
                       zona_clase=lineas[i].zona_clase, confs=[round(float(v), 4) for v in confs][: len(t)],
                       zona=lineas[i].zona, tipo=lineas[i].tipo)
            (bordes if borde else nucleo).append(tz)
        n0 = sum(ch.isalnum() for ch in lineas[i].texto)
        n1 = sum(ch.isalnum() for tz in nucleo for ch in tz.texto)
        # se acepta si las bandas centrales leen bastante más que la caja entera (un renglón de verdad cortado por la
        # banda de las minúsculas da trozos cortos o de poca confianza) y con buena confianza
        if len(nucleo) >= 2 and n1 >= 1.4 * max(n0, 1) and min(tz.conf for tz in nucleo) >= 0.7 and np.mean([tz.conf for tz in nucleo]) >= 0.8:
            pass
        elif len(nucleo) == 1 and bordes and _parecido(nucleo[0].texto, lineas[i].texto) >= 0.6:
            pass  # una banda central (la que ya se leyó) y restos en el borde
        else:
            continue
        # bordes: solo si no repiten un renglón vecino ya leído
        x0, _, x1, _ = lineas[i].caja
        for tz in bordes:
            if sum(ch.isalnum() for ch in tz.texto) < 4 or tz.conf < 0.85:
                continue
            yc = (tz.caja[1] + tz.caja[3]) / 2
            hb = tz.caja[3] - tz.caja[1]
            vecinos = [lineas[j].texto for j in range(len(lineas)) if j != i and abs(cy[j] - yc) < 1.6 * hb
                       and min(C[j, 2], x1) - max(C[j, 0], x0) > 0.3 * (x1 - x0)]
            if all(_parecido(tz.texto, v) < 0.6 for v in vecinos) and all(_parecido(tz.texto, n.texto) < 0.6 for n in nucleo):
                nucleo.append(tz)
        if len(nucleo) >= 2:
            nucleo.sort(key=lambda tz: tz.caja[1])
            nuevas[i] = nucleo
            n_partidas += 1
    out, nuevo = [], []
    for i, l in enumerate(lineas):
        for tz in nuevas.get(i, [l]):
            out.append(tz)
            nuevo.append(i in nuevas)
    return deduplicar(out, nuevo), n_partidas


def deduplicar(lineas: list, nuevo: list | None = None) -> list:
    """Quita renglones repetidos (misma altura, mismas columnas y texto parecido): al partir una caja doble, uno de sus
    trozos puede ser un renglón que ya tenía caja propia. Se queda el original o, entre dos nuevos, el más seguro."""
    n = len(lineas)
    if n < 2:
        return lineas
    nuevo = nuevo or [False] * n
    C = np.array([l.caja for l in lineas], float)
    hh = C[:, 3] - C[:, 1]
    cy = (C[:, 1] + C[:, 3]) / 2
    fuera = set()
    for i in range(n):
        if not nuevo[i] or i in fuera:
            continue
        cerca = np.nonzero((np.abs(cy - cy[i]) < 0.6 * np.minimum(hh, hh[i]))
                           & ((np.minimum(C[:, 2], C[i, 2]) - np.maximum(C[:, 0], C[i, 0])) > 0.5 * np.minimum(C[:, 2] - C[:, 0], C[i, 2] - C[i, 0])))[0]
        for j in cerca:
            if j == i or j in fuera or _parecido(lineas[i].texto, lineas[j].texto) < 0.7:
                continue
            if not nuevo[j]:
                fuera.add(i)
                break
            fuera.add(i if lineas[i].conf < lineas[j].conf else j)
            if i in fuera:
                break
    return [l for k, l in enumerate(lineas) if k not in fuera]


# ---------------------------------------------------------------- 2. renglones que cruzan un canal
def partir_por_canales(lineas: list, ventana: float = 10.0, min_lados: int = 3) -> tuple[list, int]:
    if len(lineas) < 10:
        return lineas, 0
    C = np.array([l.caja for l in lineas], float)
    h = _alto_cuerpo(C, [l.texto for l in lineas])
    cy = (C[:, 1] + C[:, 3]) / 2
    alto = C[:, 3] - C[:, 1]
    out, n = [], 0
    for i, l in enumerate(lineas):
        x0, y0, x1, y1 = l.caja
        t, xs = l.texto, l.xs
        if x1 - x0 < 6 * h or not xs or len(xs) != len(t) or len(t) < 6:
            out.append(l)
            continue
        cerca = (np.abs(cy - cy[i]) < ventana * h) & (np.arange(len(C)) != i) & (alto < 2.0 * h)
        V = C[cerca]
        if len(V) < 2 * min_lados:
            out.append(l)
            continue
        cortes = []
        for k in range(1, len(t)):
            hueco = xs[k] - xs[k - 1]
            if not (t[k - 1] == " " or t[k] == " " or hueco > 0.9 * h):
                continue
            x = (xs[k - 1] + xs[k]) / 2
            if x - x0 < 2.5 * h or x1 - x < 2.5 * h:
                continue
            cruzan = int(((V[:, 0] < x - 0.4 * h) & (V[:, 2] > x + 0.4 * h)).sum())
            acaban = int(((V[:, 2] <= x + 0.6 * h) & (V[:, 2] > x - 4 * h)).sum())
            empiezan = int(((V[:, 0] >= x - 0.6 * h) & (V[:, 0] < x + 4 * h)).sum())
            if acaban >= min_lados and empiezan >= min_lados and cruzan <= max(1, 0.1 * (acaban + empiezan)):
                cortes.append((-(acaban + empiezan) + 2 * cruzan - 0.01 * hueco, x, k))
        if not cortes:
            out.append(l)
            continue
        elegidos = []
        for _, x, k in sorted(cortes):
            if all(abs(x - e[0]) > 2.5 * h for e in elegidos):
                elegidos.append((x, k))
        elegidos.sort()
        confs = l.confs if len(l.confs) == len(t) else [l.conf] * len(t)
        ini, x_ini, trozos = 0, x0, []
        for x, k in elegidos + [(x1, len(t))]:
            seg = t[ini:k]
            a, b = len(seg) - len(seg.lstrip()), len(seg.rstrip())
            if seg.strip():
                k0, k1 = ini + a, ini + b
                cx0 = max(x_ini, xs[k0] - 0.6 * h)
                cx1 = min(x, xs[k1 - 1] + 0.6 * h)
                cf = confs[k0:k1]
                trozos.append(Linea([round(cx0, 1), y0, round(cx1, 1), y1], t[k0:k1], round(float(np.mean(cf)), 4) if cf else l.conf,
                                    xs[k0:k1], zona_clase=l.zona_clase, confs=cf, zona=l.zona, tipo=l.tipo, estilos=[]))
            ini, x_ini = k, x
        if len(trozos) > 1:
            n += len(trozos) - 1
            out.extend(trozos)
        else:
            out.append(l)
    return out, n


# ---------------------------------------------------------------- 3. orden de Breuel
def orden_breuel(cajas, h: float | None = None) -> list[int]:
    n = len(cajas)
    if n <= 1:
        return list(range(n))
    B = np.asarray(cajas, float)
    if h is None:
        h = float(np.median(B[:, 3] - B[:, 1]))
    tol = 0.25 * h
    x0, y0, x1, y1 = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
    cy = (y0 + y1) / 2
    # solape en x relativo al más estrecho de los dos: las cajas de una columna invaden a veces unos píxeles la vecina
    w = np.maximum(x1 - x0, 1.0)
    ov = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    sx = (ov > tol) & (ov > 0.2 * np.minimum(w[:, None], w[None, :]))
    r1 = sx & (cy[:, None] < cy[None, :])
    cx = (x0 + x1) / 2
    izq = ~sx & (cx[:, None] < cx[None, :])
    r2 = np.zeros((n, n), bool)
    ia, ib = np.nonzero(izq)
    if len(ia):
        ylo, yhi = np.minimum(cy[ia], cy[ib]), np.maximum(cy[ia], cy[ib])
        res = np.empty(len(ia), bool)
        lote = max(1, 4_000_000 // n)
        for s in range(0, len(ia), lote):
            a_, b_ = ia[s:s + lote], ib[s:s + lote]
            c_ok = sx[a_] & sx[b_] & (cy[None, :] > ylo[s:s + lote, None]) & (cy[None, :] < yhi[s:s + lote, None])
            res[s:s + lote] = ~c_ok.any(1)
        r2[ia[res], ib[res]] = True
    antes = r1 | r2
    np.fill_diagonal(antes, False)
    pend = antes.sum(0).astype(int)
    hecho = np.zeros(n, bool)
    # desempate entre renglones sin precedencia: el de más arriba (y luego el de más a la izquierda)
    rango = np.empty(n, int)
    rango[np.lexsort((x0, np.round(y0 / max(h, 1.0))))] = np.arange(n)
    out = []
    for _ in range(n):
        libres = np.nonzero((pend == 0) & ~hecho)[0]
        if len(libres) == 0:  # ciclo (raro): el que tenga menos predecesores pendientes
            cand = np.nonzero(~hecho)[0]
            libres = cand[pend[cand] == pend[cand].min()]
        i = int(libres[np.argmin(rango[libres])])
        hecho[i] = True
        out.append(i)
        pend[antes[i]] -= 1
    return out


def _mayus(t: str) -> bool:
    letras = [c for c in t if c.isalpha()]
    return bool(letras) and sum(c.isupper() for c in letras) >= 0.8 * len(letras)


def grupos_de_fila(ls, C: np.ndarray, h: float) -> list[list[int]]:
    """Trozos de un mismo título partido en varias cajas (la maqueta los pone en la misma zona; misma altura, mismo
    cuerpo y poco hueco entre ellos) -> grupos que se ordenan como una sola caja, de izquierda a derecha."""
    n = len(ls)
    padre = list(range(n))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    alto = C[:, 3] - C[:, 1]
    grande = alto >= 1.5 * h  # solo letra de titular: en el cuerpo, la misma zona de la maqueta mezcla columnas
    for i in range(n):
        if ls[i].zona < 0 or not grande[i]:
            continue
        for j in range(i + 1, n):
            if ls[j].zona != ls[i].zona or not grande[j]:
                continue
            hi, hj = alto[i], alto[j]
            if max(hi, hj) > 1.25 * min(hi, hj):
                continue
            sy = min(C[i, 3], C[j, 3]) - max(C[i, 1], C[j, 1])
            if sy < 0.7 * min(hi, hj):
                continue
            hueco = max(C[i, 0], C[j, 0]) - min(C[i, 2], C[j, 2])
            # y con el mismo estilo de letra (dos titulares vecinos de columnas distintas suelen diferir: VERSALES
            # frente a minúsculas)
            if hueco < 2.0 * max(hi, hj) and _mayus(ls[i].texto) == _mayus(ls[j].texto):
                padre[raiz(i)] = raiz(j)
    g = {}
    for i in range(n):
        g.setdefault(raiz(i), []).append(i)
    return [sorted(v, key=lambda i: C[i, 0]) for v in g.values()]


def orden_por_grupos(ls, C: np.ndarray, h: float) -> list[int]:
    gs = grupos_de_fila(ls, C, h)
    cajas = np.array([[C[g, 0].min(), C[g, 1].min(), C[g, 2].max(), C[g, 3].max()] for g in gs], float)
    return [i for k in orden_breuel(cajas, h) for i in gs[k]]


# ---------------------------------------------------------------- 4. bloques
def bloques(lineas: list, h: float) -> list[list[int]]:
    out, act = [], []
    for i, l in enumerate(lineas):
        if act:
            p, c = lineas[act[-1]].caja, l.caja
            solape = min(p[2], c[2]) - max(p[0], c[0])
            hp, hc = p[3] - p[1], c[3] - c[1]
            # mismo cuerpo de letra: solo cuenta la diferencia si uno de los dos es de titular (las cajas de los
            # renglones releídos son más bajas que las del detector, y se solapan más)
            titular = max(hp, hc) > 1.45 * h and max(hp, hc) > 1.4 * min(hp, hc)
            if not (solape > 0.5 * min(p[2] - p[0], c[2] - c[0]) and -1.0 * h < c[1] - p[3] < 1.2 * h and not titular):
                out.append(act)
                act = []
        act.append(i)
    if act:
        out.append(act)
    return out


def _tipo(p: Pagina, l) -> str | None:
    if 0 <= l.zona < len(p.zonas):
        z = p.zonas[l.zona]
        return z.get("tipo") or z.get("clase")
    return l.zona_clase


def procesar(p: Pagina, img: Image.Image, completo=None) -> dict:
    """Aplica los cuatro pasos sobre la página (in situ). -> resumen."""
    def aparte(l):  # renglones en figuras (rótulos, ruido de las fotos) y vacíos: conservan su zona y van al final
        return bool(l.en_figura) or not l.texto.strip() or _tipo(p, l) in NO_TEXTO

    fuera = [l for l in p.lineas if aparte(l)]
    ls = [l for l in p.lineas if not aparte(l)]
    if len(ls) < 10:
        return {}
    info = {}
    if os.environ.get("AMANUENSIS_PRENSA_DOBLES", "1") != "0":
        ls, info["dobles"] = partir_dobles(ls, img, completo)
    if os.environ.get("AMANUENSIS_PRENSA_CANALES", "1") != "0":
        ls, info["canales"] = partir_por_canales(ls)
    C = np.array([l.caja for l in ls], float)
    h = _alto_cuerpo(C, [l.texto for l in ls])
    o = orden_por_grupos(ls, C, h)
    ls = [ls[i] for i in o]
    bl = bloques(ls, h)
    # zonas: se quedan las que no son de texto (figuras, tablas…) y las de los renglones apartados; las de texto se
    # sustituyen por un bloque de columna cada una
    tipos_l = [_tipo(p, l) for l in ls]
    guardar = {i for i, z in enumerate(p.zonas) if (z.get("tipo") or z.get("clase")) in NO_TEXTO}
    guardar |= {l.zona for l in fuera if 0 <= l.zona < len(p.zonas)}
    guardar |= {l.zona for l, t in zip(ls, tipos_l) if t in TABLA and 0 <= l.zona < len(p.zonas)}
    mapa = {}
    zonas = []
    for i, z in enumerate(p.zonas):
        if i in guardar:
            mapa[i] = len(zonas)
            zonas.append(z)
    for l in fuera:
        l.zona = mapa.get(l.zona, -1)
    for l, t in zip(ls, tipos_l):
        l._tipo_prensa = t
        if t in TABLA:
            l.zona = mapa.get(l.zona, -1)
    for b in bl:
        tipos = [ls[i]._tipo_prensa for i in b if ls[i]._tipo_prensa and ls[i]._tipo_prensa not in TABLA]
        if not tipos:
            continue
        tipo = max(set(tipos), key=tipos.count)
        if tipo in NO_TEXTO:
            tipo = "MainZone"
        cj = [min(ls[i].caja[0] for i in b), min(ls[i].caja[1] for i in b), max(ls[i].caja[2] for i in b), max(ls[i].caja[3] for i in b)]
        zonas.append({"clase": -1, "tipo": tipo, "caja": [round(v, 1) for v in cj], "conf": 1.0, "prensa": True})
        for i in b:
            if ls[i]._tipo_prensa not in TABLA:
                ls[i].zona = len(zonas) - 1
                ls[i].zona_clase = tipo
    for l in ls:
        del l._tipo_prensa
    p.zonas = zonas
    p.lineas = ls + fuera
    info["bloques"] = len(bl)
    return info
