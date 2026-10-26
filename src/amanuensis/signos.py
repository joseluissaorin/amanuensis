"""Los signos que el lector confunde porque se parecen: se deciden midiendo la imagen o por la notación del documento.

- Raya media (–) y raya larga (—): para el lector son el mismo trazo con distinto largo. Se mide el trazo en la imagen,
  en relación con la altura del renglón: la media de los incisos y de las listas mide de 0,3 a 0,5 cuerpos; la larga
  del diálogo, casi uno. Los guiones no se tocan (su medida es ruidosa y casi nunca se confunden).
- ⊢ (lógica: «entraña») no está en el juego de caracteres de Ligero, que escribe «—» en su lugar. Se reconoce por
  su forma: una raya con un palo vertical en el extremo izquierdo, suelta entre espacios.
- Notación propia del documento («+>» implica, «>>» presupone): si un signo aparece muchas veces bien leído, sus
  lecturas rotas («»>», «+'») se devuelven a él, como los nombres de personaje.

Calibrado en «Las implicaturas conversacionales» (Pragmática para hispanistas): ⊢ palo 0,55-0,65 y ancho 0,55-0,69;
raya media 0,33-0,46; raya larga 0,93.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

RAYAS = "–—"
MEDIA = 0.28  # por debajo, la medida no es fiable (el lector puso el centro fuera del trazo): no se cambia
PALO = 0.5  # alto del palo vertical / alto del renglón a partir del cual la raya es un ⊢


def medir_raya(g: np.ndarray, caja, xc: float):
    """(ancho / alto, palo izquierdo / alto) del trazo horizontal que pasa por xc en el renglón `caja`; None si no hay
    tinta. `g`: la página en gris (uint8)."""
    x0, y0, x1, y1 = [int(round(v)) for v in caja[:4]]
    h = max(6, y1 - y0)
    X0, X1 = int(max(0, xc - 1.4 * h)), int(min(g.shape[1], xc + 1.4 * h))
    Y0, Y1 = int(max(0, y0 - 0.15 * h)), int(min(g.shape[0], y1 + 0.15 * h))
    a = g[Y0:Y1, X0:X1]
    if a.size == 0:
        return None
    tinta = a < min((int(a.max()) + int(a.min())) / 2, 160)
    c = min(max(0, int(xc) - X0), tinta.shape[1] - 1)
    ancho, izq = 0, None
    for y in range(int(0.25 * h), min(int(0.85 * h), tinta.shape[0])):
        if not tinta[y, c]:
            continue
        fila = tinta[y]
        l = c
        while l > 0 and fila[l - 1]:
            l -= 1
        r = c
        while r + 1 < len(fila) and fila[r + 1]:
            r += 1
        if r - l > ancho:
            ancho, izq = r - l, l
    if not ancho:
        return None
    palo = int(tinta[:, max(0, izq - 1):izq + 2].any(1).sum())
    return ancho / h, palo / h


def medir_rayas(g: np.ndarray, linea) -> None:
    """Mide en la imagen cada raya del renglón y lo guarda en `linea._rayas` [(posición, ancho, palo)]; la decisión es
    del documento entero (decidir_rayas), porque cada fuente tiene sus anchos."""
    t = linea.texto
    if not any(c in RAYAS for c in t) or not linea.xs or len(linea.xs) != len(t):
        return
    medidas = []
    for k, ch in enumerate(t):
        if ch in RAYAS:
            m = medir_raya(g, linea.caja, linea.xs[k])
            if m is not None:
                medidas.append((k, m[0], m[1]))
    if medidas:
        linea._rayas = medidas


def _dos_grupos(v: list[float]):
    """Umbral entre dos grupos de anchos si los hay (hueco claro entre el corto y el largo), o None."""
    v = sorted(v)
    if len(v) < 4:
        return None
    huecos = [(v[i + 1] - v[i], i) for i in range(len(v) - 1)]
    hueco, i = max(huecos)
    cortos, largos = v[: i + 1], v[i + 1:]
    if hueco < 0.18 or min(len(cortos), len(largos)) < 1 or largos[0] < 1.4 * cortos[-1]:
        return None
    return (cortos[-1] + largos[0]) / 2


def decidir_rayas(paginas) -> None:
    """Con las medidas de todo el documento: ⊢ por su forma (palo vertical a la izquierda, suelta entre espacios); y
    raya media o larga solo si sus anchos forman dos grupos (el corto es la media). Con un grupo solo, se respeta lo
    que leyó el modelo. Misma longitud: las confianzas, las x y los fotogramas siguen valiendo."""
    todas = [(l, k, a, p) for pg in paginas for l in pg.lineas for k, a, p in getattr(l, "_rayas", [])]
    if not todas:
        return
    corte = _dos_grupos([a for _, _, a, p in todas if p < PALO and a >= MEDIA])
    for l, k, a, p in todas:
        t = l.texto
        if k >= len(t) or t[k] not in RAYAS:
            continue
        suelta = (k == 0 or t[k - 1] == " ") and (k + 1 == len(t) or t[k + 1] == " ")
        nuevo = t[k]
        if p >= PALO and suelta and k > 0:
            nuevo = "⊢"
        elif corte is not None and a >= MEDIA:
            nuevo = "—" if a >= corte else "–"
        if nuevo != t[k]:
            l.texto = t[:k] + nuevo + t[k + 1:]
    for _, pg in enumerate(paginas):
        for l in pg.lineas:
            if hasattr(l, "_rayas"):
                del l._rayas


# ---------------------------------------------------------------- notación del documento
# (no actúan tras un «+», que es el «+ >» o «++ >» de Levinson, ni cerrando una escala de Horn «<TODOS, ALGUNOS>»)
NOTACION = {"+>": [(r"\+(?=[‘'’])", "+> "), (r"\+>(?=[‘'’\w])", "+> ")],
            ">>": [(r"»>", ">>"), (r"(?<=[^\s+] )>(?=\s+[‘'’])", ">>"), (r"(?<=[^\s+] )>(?=[‘'’])", ">> ")]}


def notacion(paginas) -> None:
    """Si el documento usa «+>» o «>>» al menos cinco veces bien leídos, sus lecturas rotas vuelven a ellos: «(+‘No»
    pasa a «(+> ‘No», «+>No» a «+> No», «»>» a «>>» y «basura> ‘Beatriz» a «basura >> ‘Beatriz»."""
    texto = "\n".join(l.texto for p in paginas for l in p.lineas)
    n = Counter({k: len(re.findall(re.escape(k) + r"\s", texto)) for k in NOTACION})
    for signo, rotos in NOTACION.items():
        if n[signo] < 5:
            continue
        for p in paginas:
            for l in p.lineas:
                t = l.texto
                if signo == ">>" and "<" in t:
                    continue  # una escala de Horn en el renglón: sus «>» son cierres
                for patron, por in rotos:
                    t = re.sub(patron, por, t)
                t = re.sub(r"  +", " ", t)
                if t != l.texto:
                    l.texto = t
                    l.confs = []


# ---------------------------------------------------------------- comillas por parejas
ABRE = {"‘": "’", "“": "”", "«": "»"}
CIERRA = {v: k for k, v in ABRE.items()}
RECTAS = "'\""
PAREJA_UMBRAL = 0.01  # con la pareja ya escrita, basta esta probabilidad en el hueco


def _es_apertura(t: str, i: int) -> bool:
    c = t[i]
    if c in ABRE:
        return True
    if c in RECTAS:  # recta: abre si va tras espacio o principio y delante de letra
        return (i == 0 or not t[i - 1].isalnum()) and i + 1 < len(t) and t[i + 1].isalnum()
    return False


def _es_cierre(t: str, i: int) -> bool:
    c = t[i]
    if c in CIERRA:
        return True
    if c in RECTAS:
        return i > 0 and (t[i - 1].isalnum() or t[i - 1] in ".?!") and (i + 1 == len(t) or not t[i + 1].isalnum())
    return False


def _insertar(l, i: int, ch: str, fotograma: int, p: float) -> None:
    lp, fot, chars = l._ctc
    l.texto = l.texto[:i] + ch + l.texto[i:]
    fot.insert(i, fotograma)
    if l.confs and len(l.confs) == len(l.texto) - 1:
        l.confs.insert(i, round(p, 4))
    if l.xs and len(l.xs) == len(l.texto) - 1:
        l.xs.insert(i, l.xs[i - 1] if i > 0 else l.xs[0])


def comillas_pareadas(paginas) -> int:
    """Una comilla sin pareja en su renglón (de cierre sin apertura delante, o de apertura sin cierre detrás) tiene su
    pareja en el hueco entre palabras donde el lector le da más probabilidad, aunque no ganara al blanco: «Ser
    simpática’» → «‘Ser simpática’». Y una pareja con formas distintas («“Voy … cita’») toma la forma del cierre.
    Solo en renglones con la CTC alineada (antes de la corrección léxica, que puede cambiar longitudes)."""
    n = 0
    for p in paginas:
        for l in p.lineas:
            ctc = getattr(l, "_ctc", None)
            if ctc is None or len(ctc[1]) != len(l.texto):
                continue
            lp, fot, chars = ctc
            cand = [i + 1 for i, c in enumerate(chars) if c in "‘’'“”\""]
            if not cand:
                continue
            prob = np.exp(lp[:, cand]).max(1)
            t = l.texto
            aperturas = [i for i in range(len(t)) if _es_apertura(t, i)]
            cierres = [i for i in range(len(t)) if _es_cierre(t, i) and i not in aperturas]
            # cierre sin apertura delante (en el renglón)
            for c in cierres:
                if any(a < c for a in aperturas):
                    continue
                huecos = [m.start() for m in re.finditer(r"(?<![\w’'”])(?=\w)", t[:c])]
                huecos = [h for h in huecos if h > 0 and t[h - 1] == " " or h == 0]
                if not huecos:
                    continue
                def p_hueco(h):
                    f0 = fot[h - 1] + 1 if h > 0 else 0
                    f1 = fot[h]
                    return (float(prob[f0:f1].max()), f0 + int(prob[f0:f1].argmax())) if f1 > f0 else (0.0, f0)
                h = max(huecos, key=lambda h: p_hueco(h)[0])
                pv, fr = p_hueco(h)
                if pv >= PAREJA_UMBRAL:
                    ap = {"’": "‘", "”": "“", "»": "«"}.get(t[c], t[c])
                    _insertar(l, h, ap, fr, pv)
                    n += 1
                    break  # un arreglo por renglón: los índices han cambiado
            else:
                # apertura sin cierre detrás: el cierre, en el hueco tras una palabra con más probabilidad
                for a in aperturas:
                    if any(c > a for c in cierres):
                        continue
                    finales = [m.end() for m in re.finditer(r"\w(?=[\s),;:.]|$)", t[a:])]
                    finales = [a + e for e in finales]
                    if not finales:
                        continue
                    def p_fin(e):
                        f0 = fot[e - 1] + 1
                        f1 = fot[e] if e < len(fot) else lp.shape[0]
                        return (float(prob[f0:f1].max()), f0 + int(prob[f0:f1].argmax())) if f1 > f0 else (0.0, f0)
                    e = max(finales, key=lambda e: p_fin(e)[0])
                    pv, fr = p_fin(e)
                    if pv >= 2 * PAREJA_UMBRAL:
                        ci = ABRE.get(t[a], t[a])
                        _insertar(l, e, ci, fr, pv)
                        n += 1
                    break
            # parejas con formas distintas: la apertura toma la forma del cierre
            t = l.texto
            ap = [i for i in range(len(t)) if t[i] in ABRE]
            for i in ap:
                j = next((k for k in range(i + 1, len(t)) if t[k] in CIERRA or t[k] in ABRE), None)
                if j is not None and t[j] in CIERRA and CIERRA[t[j]] != t[i]:
                    l.texto = t[:i] + CIERRA[t[j]] + t[i + 1:]
                    t = l.texto
    return n
