"""Columnas de prensa y revista: las cajas de renglón que cruzan el canal entre dos columnas se parten.

El detector Mínimo mira la página reducida a 768 píxeles de ancho; a esa escala el canal estrecho de una revista a tres
columnas desaparece y dos renglones a la misma altura salen como una sola caja («…algunos de los profesores eran por
ejemplo puede decir: “es solamente esto”…»). Ningún orden de lectura arregla eso después.

Aquí se usa lo que mira un maquetador: los renglones de una columna empiezan todos en la misma x. Esos comienzos
alineados dan las columnas; una caja que pasa por encima del comienzo de otra columna se parte por la franja blanca
vertical que hay justo antes, buscada en la imagen a resolución completa.
"""

from __future__ import annotations

import numpy as np

MIN_RENGLONES = 4  # renglones que tienen que empezar en una x para que sea el comienzo de una columna
ANCHO_MIN = 8  # cuerpos como mínimo entre los comienzos de dos columnas
PARTE_MIN = 0.15  # fracción de los renglones largos de la página que tiene que empezar en una columna
HUECO_TROZOS = 3.0  # cuerpos de hueco como mucho entre dos trozos del mismo renglón (el detector deja texto sin cubrir)


def comienzos(cajas, h: float) -> list[float]:
    """x donde empiezan las columnas: comienzos de renglón (de renglones largos) alineados a menos de un cuerpo."""
    xs = sorted(c[0] for c in cajas if c[2] - c[0] > 4 * h)
    grupos, act = [], []
    for x in xs:
        if act and x - act[-1] > 0.8 * h:
            grupos.append(act)
            act = []
        act.append(x)
    if act:
        grupos.append(act)
    # una columna de verdad reúne una buena parte de los renglones largos (un esquema o un recuadro, no)
    minimo = max(MIN_RENGLONES, PARTE_MIN * len(xs))
    starts = [float(np.median(g)) for g in grupos if len(g) >= minimo]
    # una sangría de párrafo no es una columna: dos comienzos de columna distan al menos ANCHO_MIN cuerpos
    out = []
    for x in starts:
        if not out or x - out[-1] >= ANCHO_MIN * h:
            out.append(x)
    return out


def _blanco(g: np.ndarray, caja, x0: int, x1: int, h: float):
    """La franja blanca vertical más ancha de la caja entre x0 y x1: (inicio, fin) o None."""
    y0, y1 = int(max(0, caja[1])), int(min(g.shape[0], caja[3]))
    x0, x1 = int(max(0, x0)), int(min(g.shape[1], x1))
    if y1 - y0 < 3 or x1 - x0 < 3:
        return None
    a = g[y0:y1, x0:x1]
    umbral = min(160, (int(a.max()) + int(a.min())) // 2)
    tinta = (a < umbral).mean(0)
    blanco = tinta <= 0.02
    mejor, ini = None, None
    for i, b in enumerate(list(blanco) + [False]):
        if b and ini is None:
            ini = i
        elif not b and ini is not None:
            if mejor is None or i - ini > mejor[1] - mejor[0]:
                mejor = (ini, i)
            ini = None
    if mejor is None or mejor[1] - mejor[0] < 0.25 * h:
        return None
    return x0 + mejor[0], x0 + mejor[1]


def partir_por_canales(img, cajas: list) -> tuple[list, int]:
    """Parte las cajas que cruzan el comienzo de otra columna por la franja blanca del canal. -> (cajas, cuántas)."""
    if len(cajas) < 2 * MIN_RENGLONES:
        return cajas, 0
    h = float(np.median([c[3] - c[1] for c in cajas]))
    cols = comienzos(cajas, h)
    if len(cols) < 2:
        return cajas, 0
    g = np.asarray(img.convert("L"))
    out, n = [], 0
    for c in cajas:
        trozos = [list(c)]
        for s in cols:
            nuevos = []
            for t in trozos:
                if t[0] + 2 * h < s < t[2] - 2 * h:
                    b = _blanco(g, t, s - 1.6 * h, s + 0.3 * h, h)
                    if b is not None:
                        nuevos += [[t[0], t[1], float(b[0]), t[3]], [float(b[1]), t[1], t[2], t[3]]]
                        n += 1
                        continue
                nuevos.append(t)
            trozos = nuevos
        out += trozos
    return out, n


def _hueco_con_tinta(g: np.ndarray, c, d, h: float) -> bool:
    """¿Hay texto sin cubrir entre las cajas c y d (misma fila)? Un hueco blanco más ancho que un espacio entre
    palabras es un canal entre columnas, no un trozo perdido."""
    x0, x1 = int(c[2]), int(d[0])
    if x1 - x0 <= 0.6 * h:
        return True  # un espacio entre palabras: se une
    y0, y1 = int(max(c[1], d[1])), int(min(c[3], d[3]))
    if y1 - y0 < 3:
        return False
    a = g[y0:y1, max(0, x0):min(g.shape[1], x1)]
    if a.size == 0:
        return False
    umbral = min(160, (int(g[y0:y1].max()) + int(g[y0:y1].min())) // 2)
    columnas_con_tinta = ((a < umbral).mean(0) > 0.03).mean()
    return columnas_con_tinta > 0.35


def unir_trozos(cajas: list, cols: list[float], h: float, g: np.ndarray | None = None) -> list:
    """Une los trozos de un mismo renglón que el detector partió (un subrayado a lápiz, una mancha), y con ellos el
    texto que dejó sin cubrir entre medias: misma fila, a menos de HUECO_TROZOS cuerpos y sin un comienzo de columna
    entre medias."""
    orden = sorted(range(len(cajas)), key=lambda i: (cajas[i][1], cajas[i][0]))
    usadas, out = set(), []
    for a in orden:
        if a in usadas:
            continue
        c = list(cajas[a])
        usadas.add(a)
        cambio = True
        while cambio:
            cambio = False
            for b in orden:
                if b in usadas:
                    continue
                d = cajas[b]
                hm = min(c[3] - c[1], d[3] - d[1])
                solape = min(c[3], d[3]) - max(c[1], d[1])
                hueco = d[0] - c[2]
                if solape > 0.4 * hm and hueco < HUECO_TROZOS * h and d[2] > c[0] \
                        and not any(min(c[2], d[0]) - 0.3 * h < s < d[0] + 0.8 * h for s in cols if d[0] > c[0] + 2 * h) \
                        and (g is None or _hueco_con_tinta(g, c, d, h)):
                    c = [min(c[0], d[0]), min(c[1], d[1]), max(c[2], d[2]), max(c[3], d[3])]
                    usadas.add(b)
                    cambio = True
        out.append(c)
    return out


def maquetar(img, cajas: list) -> tuple[list, dict]:
    """Columnas de prensa y revista: une los trozos de renglón, parte las cajas que cruzan un canal y, si la página tiene
    dos columnas o más, las ordena por columnas (corte XY con canales). -> (cajas en orden, resumen)."""
    from .detector import orden_lectura

    if len(cajas) < 2 * MIN_RENGLONES:
        return cajas, {}
    h = float(np.median([c[3] - c[1] for c in cajas]))
    cols = comienzos(cajas, h)
    if len(cols) < 2:
        return cajas, {}
    n0 = len(cajas)
    cajas = unir_trozos(cajas, cols, h, np.asarray(img.convert("L")))
    unidas = n0 - len(cajas)
    cajas, partidas = partir_por_canales(img, cajas)
    # el orden, sobre las cajas ya unidas y partidas: por grafo (orden_gcn.py) si están sus pesos, y si no por columnas
    # (Máscara, revista a tres columnas: WER 12,6 -> 6,2 %; sin cambios en los libros a una columna, que no llegan aquí)
    from .detector import modo_orden

    orden = orden_lectura(cajas, {"columnas": "columnas", "ligero": "aprendido"}.get(modo_orden(), "gcn"), img.size)
    return [cajas[i] for i in orden], {"columnas": len(cols), "unidas": unidas, "partidas": partidas}


# ---------------------------------------------------------------- el texto dice qué bloque sigue
def _bloques(lineas, h: float) -> list[list[int]]:
    """Índices del orden de lectura partidos en bloques de columna: renglones seguidos que se apilan (el siguiente
    empieza debajo, a menos de dos cuerpos, y en la misma franja horizontal)."""
    bloques, act = [], []
    for i, l in enumerate(lineas):
        if act:
            a = lineas[act[-1]]
            debajo = 0 <= l.caja[1] - a.caja[3] < 2 * h or abs(l.caja[1] - a.caja[1]) < 0.3 * h
            mismo = min(a.caja[2], l.caja[2]) - max(a.caja[0], l.caja[0]) > 0.3 * (a.caja[2] - a.caja[0])
            if not (debajo and mismo):
                bloques.append(act)
                act = []
        act.append(i)
    if act:
        bloques.append(act)
    return bloques


def por_continuidad(pagina, frecuencia, minima: int = 20) -> int:
    """Si un bloque de columna acaba en palabra partida («para el comba-») y otro empieza por el resto de esa palabra
    («te). En la posición») y la palabra unida existe, el segundo va justo detrás del primero. -> bloques movidos."""
    ls = pagina.lineas
    if len(ls) < 6:
        return 0
    h = float(np.median([l.caja[3] - l.caja[1] for l in ls]))
    if len(comienzos([l.caja for l in ls], h)) < 2:
        return 0  # a una columna, un bloque que no sigue a la palabra partida es una figura o un esquema, no otra columna
    bloques = _bloques(ls, h)
    if len(bloques) < 3:
        return 0
    import re

    def cabeza(b) -> bool:
        """¿El bloque b empieza arriba de una columna? Si tiene encima un renglón de su misma franja (a menos de dos
        cuerpos) o a su izquierda un trozo de su mismo renglón, es la continuación de ese renglón y no se mueve
        (Máscara, p. 63, con 0,2° de inclinación residual: «estabiliz» | «ción, inmovilidad» no es otra columna)."""
        c = ls[bloques[b][0]].caja
        for i, l in enumerate(ls):
            if i in bloques[b]:
                continue
            d = l.caja
            solape_x = min(c[2], d[2]) - max(c[0], d[0])
            if solape_x > 0.3 * min(c[2] - c[0], d[2] - d[0]) and 0 <= c[1] - d[3] < 2 * h and d[1] < c[1]:
                return False
            solape_y = min(c[3], d[3]) - max(c[1], d[1])
            if solape_y > 0.5 * min(c[3] - c[1], d[3] - d[1]) and 0 <= c[0] - d[2] < 3 * h:
                return False
        return True

    def una(a, b) -> int:
        """Frecuencia de la palabra que forman el trozo partido al final del bloque a y el principio del bloque b (0 si
        no la hay o es rara)."""
        m = re.search(r"([^\W\d_]+)[-¬]$", ls[bloques[a][-1]].texto.rstrip())
        t = ls[bloques[b][0]].texto.lstrip()
        n = re.match(r"[^\W\d_]+", t)
        if m is None or n is None or not t[:1].islower() or len(m.group(1)) < 2 or len(n.group(0)) < 2:
            return 0  # (un trozo de una letra es ruido del detector: «Oriente-» + «m», «te-» + «a»)
        f = frecuencia((m.group(1) + n.group(0)).lower())
        return f if f >= minima else 0

    def libre(b) -> bool:
        """El bloque b se puede mover: empieza arriba de una columna y no sigue ya a la palabra partida que lo completa
        (Máscara, p. 63, a 1,5° de más: «comba-» | «te).» ya unidos no se rompen por «ac-» + «te» al final de la página)."""
        return cabeza(b) and not (b > 0 and una(b - 1, b))

    movidos = 0
    k = 0
    while k < len(bloques):
        if re.search(r"[^\W\d_][-¬]$", ls[bloques[k][-1]].texto.rstrip()):
            if k + 1 >= len(bloques) or not una(k, k + 1):
                cands = sorted(((una(k, b), b) for b in range(len(bloques)) if b not in (k, k + 1)), reverse=True)
                cands = [c for c in cands if c[0]]
                # la palabra más frecuente, si gana de lejos («combate» 145 161 frente a «combatan» 175), también a
                # la rareza cuando es la única («ac-» + «te», 158, o «ac-» + «en», 46, no bastan para mover un bloque)
                otra = cands[1][0] if len(cands) > 1 else minima
                if cands and cands[0][0] >= 10 * max(otra, minima) and libre(cands[0][1]) and movidos < len(bloques):
                    b = cands[0][1]
                    bloque = bloques.pop(b)
                    if b < k:
                        k -= 1  # el bloque venía de delante: el actual retrocede un puesto
                    bloques.insert(k + 1, bloque)
                    movidos += 1
        k += 1
    if movidos:
        pagina.lineas = [ls[i] for b in bloques for i in b]
    return movidos
