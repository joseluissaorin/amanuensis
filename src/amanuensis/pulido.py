"""El pulido de la página: lo que hace un lector humano con una página bien leída letra a letra pero sin maquetación.

Sin el modelo grande, Ligero lee casi todas las letras bien y pierde en la página: cabeceras corrientes y folios
metidos en el texto, el numerador «(1)» separado de su ejemplo, basura de los agujeros de la encuadernación, epígrafes
que no se reconocen como tales y romanos leídos como cifras («1) convencionales / II) no convencionales»). Aquí se
arregla con reglas sobre la geometría y el texto ya leído, sin tocar el modelo. Las clases van a `Linea.zona_clase`
con los nombres de la maquetación (RunningTitleZone, NumberingZone, Section-header, DigitizationArtefactZone): así
TEI, PAGE y Markdown las entienden sin más.
"""

from __future__ import annotations

import re
from collections import Counter

from .salidas import Linea, Pagina, es_folio, es_letras_esparcidas

VENTANA_SERIE = 25  # renglones como mucho entre dos numeradores de una misma serie (los apartados largos pasan de 12)
MARGEN = 0.12  # franja de arriba y de abajo donde viven cabeceras y folios (fracción del alto)
ENUM = re.compile(r"^\s*(\(\d{1,3}\)|\d{1,3}\)|\(?[IVXivxl1]{1,5}\)|\(?[a-z]\)|[a-z]\.|\d{1,2}\.)(\s|$)")
TITULO = re.compile(r"^\s*(\d{1,2}(\.\d{1,2})*\.?)\s+(\S.*)$")
MAYUS = "A-ZÁÉÍÓÚÜÑ"


def es_fuera(l: Linea) -> bool:
    """¿Fuera del texto de lectura? Cabecera, folio, signatura, pie de página, resto del escaneo, figura, pie de figura o
    música, por la clase de su zona (la de una maquetación de verdad o la que pone este pulido). Una línea sin clase, o
    de texto, título, lista o nota, es texto."""
    from .salidas import FW, ROL

    if not l.zona_clase:
        return False
    rol = ROL.get(l.zona_clase, "texto")
    return rol in FW or rol in ("artefacto", "figura", "pie", "musica")


def es_titulo_zona(l: Linea) -> bool:
    from .salidas import ROL

    return bool(l.zona_clase) and ROL.get(l.zona_clase, "texto") == "titulo"


def clave(t: str) -> str:
    """Una cabecera corriente sin cifras ni signos: «80 INTRODUCCIÓN A LA PRAGMÁTICA» = «INTRODUCCIÓN ... 82»."""
    return re.sub(r"[\W\d_]+", "", t).lower()


def _alto(l: Linea) -> float:
    return max(1.0, l.caja[3] - l.caja[1])


# ---------------------------------------------------------------- 1. la misma fila, un renglón
def fusionar_filas(lineas: list[Linea]) -> list[Linea]:
    """Une renglones seguidos (en orden de lectura) que están en la misma fila y cerca: el detector parte a veces el
    numerador de su texto («(1)» · «Era pobre, pero honrado») o el número del epígrafe de su título."""
    out: list[Linea] = []
    for l in lineas:
        if out:
            a = out[-1]
            h = min(_alto(a), _alto(l))
            solape = min(a.caja[3], l.caja[3]) - max(a.caja[1], l.caja[1])
            hueco = l.caja[0] - a.caja[2]
            if solape > 0.6 * h and -0.3 * h <= hueco < 3.0 * max(_alto(a), _alto(l)):
                out[-1] = _unir(a, l)
                continue
        out.append(l)
    return out


def _unir(a: Linea, b: Linea) -> Linea:
    t = a.texto.rstrip() + " " + b.texto.lstrip()
    na, nb = max(1, len(a.texto)), max(1, len(b.texto))
    return Linea([min(a.caja[0], b.caja[0]), min(a.caja[1], b.caja[1]), max(a.caja[2], b.caja[2]), max(a.caja[3], b.caja[3])],
                 t, round((a.conf * na + b.conf * nb) / (na + nb), 4),
                 xs=(a.xs + [a.caja[2]] + b.xs) if a.xs and b.xs else [],
                 confs=(a.confs + [1.0] + b.confs) if a.confs and b.confs else [])


# ---------------------------------------------------------------- 2. basura
def es_basura(l: Linea, alto: int) -> bool:
    """Restos del escaneo: los agujeros de la espiral, el borde de la hoja, una mancha. Poco seguros y sin palabras."""
    t = l.texto.strip()
    if not t:
        return True
    letras = sum(c.isalnum() for c in t)
    if ENUM.match(t) and l.conf >= 0.6:
        return False
    if letras <= 2 and l.conf < 0.6:
        return True
    fichas = [f for f in re.split(r"\s+", t) if f]
    cortas = sum(len(re.sub(r"\W", "", f)) <= 2 for f in fichas)
    # en el filo de la página (la sombra del escáner) basta una confianza floja
    filo = l.caja[1] < 0.03 * alto or l.caja[3] > 0.97 * alto
    if filo and l.conf < 0.8:
        return True
    return l.conf < 0.7 and len(fichas) >= 2 and cortas * 3 >= len(fichas) * 2


# ---------------------------------------------------------------- 3. cabeceras, folios y epígrafes
def cabeceras_repetidas(paginas: list[Pagina]) -> set[str]:
    """Claves de los renglones de la franja de arriba (o de abajo) que se repiten en varias páginas del documento."""
    n = Counter()
    for p in paginas:
        vistas = set()
        for l in p.lineas:
            if l.caja[3] <= p.alto * MARGEN or l.caja[1] >= p.alto * (1 - MARGEN):
                k = clave(l.texto)
                if len(k) >= 4:
                    vistas.add(k)
        n.update(vistas)
    minimo = 2 if len(paginas) <= 4 else 3
    return {k for k, v in n.items() if v >= minimo}


def es_cabecera(l: Linea, alto: int, repetidas: set[str]) -> str | None:
    t = l.texto.strip()
    borde = l.caja[3] <= alto * MARGEN or l.caja[1] >= alto * (1 - MARGEN)
    if not borde or len(t) > 90:
        return None
    if es_folio(t):
        return "NumberingZone"
    if clave(t) in repetidas or es_letras_esparcidas(t):
        return "RunningTitleZone"
    # «80 INTRODUCCIÓN A LA PRAGMÁTICA», «GRICE Y EL PRINCIPIO DE COOPERACIÓN 81»
    if re.fullmatch(rf"\d{{1,4}}\s+[{MAYUS}\s,.:;-]{{4,}}|[{MAYUS}\s,.:;-]{{4,}}\s+\d{{1,4}}", t):
        return "RunningTitleZone"
    return None


def es_titulo(l: Linea, sig: Linea | None, izq: float, ancho_col: float, ant: Linea | None = None) -> bool:
    """Epígrafe numerado («2.2. TIPOS DE CONTENIDOS IMPLÍCITOS», «3. Las implicaturas conversacionales») o «CAPÍTULO 5»:
    corto, no acaba en guion ni en coma y no sigue en el renglón de abajo como texto corrido."""
    t = l.texto.strip()
    if re.fullmatch(r"CAP[IÍ]TULO\s+[\dIVXLC]+\.?", t, flags=re.I):
        return True
    m = TITULO.match(t)
    if not m or len(t) > 80 or t[-1:] in "-,;" or re.match(r"^\s*\(?\d{1,3}\)", t):
        return False
    if "." not in m.group(1):  # «3 Las…»: sin punto no es epígrafe sino un número suelto
        return False
    # un número de un solo nivel («3. Las implicaturas…») se confunde con una enumeración: tiene que ir aislado
    if m.group(1).count(".") == 1 and m.group(1).endswith("."):
        h = _alto(l)
        arriba = ant is None or l.caja[1] - ant.caja[3] > 0.8 * h
        abajo = sig is None or sig.caja[1] - l.caja[3] > 0.8 * h
        if not (arriba and abajo):
            return False
    resto = m.group(3)
    if not re.match(rf"[{MAYUS}]", resto):
        return False
    # el número del epígrafe va en el margen izquierdo y el renglón no llena la columna (o es todo versalitas)
    corto = (l.caja[2] - l.caja[0]) < 0.85 * ancho_col
    versal = sum(c.isupper() for c in resto) >= 0.6 * max(1, sum(c.isalpha() for c in resto))
    return abs(l.caja[0] - izq) < 1.5 * _alto(l) and (corto or versal) and not resto.rstrip().endswith(".")


def es_versal_aislado(l: Linea, ant: Linea | None, sig: Linea | None, alto: int) -> bool:
    """Epígrafe sin número en versalitas («EN BUSCA DE REGULARIDADES», «¿CÓMO NOS COMUNICAMOS?»): todo mayúsculas, corto,
    con aire arriba y abajo, fuera de la franja de las cabeceras y sin letras espaciadas."""
    t = l.texto.strip()
    letras = [c for c in t if c.isalpha()]
    if len(letras) < 4 or len(t) > 70 or sum(c.isupper() for c in letras) < 0.85 * len(letras):
        return False
    if l.caja[3] <= alto * MARGEN or l.caja[1] >= alto * (1 - MARGEN) or es_letras_esparcidas(t) or t[-1:] in ",;-":
        return False
    h = _alto(l)
    arriba = ant is None or l.caja[1] - ant.caja[3] > 0.6 * h or ant.zona_clase in ("Section-header", "RunningTitleZone")
    abajo = sig is None or sig.caja[1] - l.caja[3] > 0.6 * h
    return arriba and abajo


# ---------------------------------------------------------------- 3b. figuras: esquemas, cuadros, gráficos
PIE = re.compile(r"^\s*(esquema|figura|fig\.|cuadro|gr[aá]fico|tabla|mapa|ilustraci[oó]n|l[aá]mina)\s+[\dIVX]+([.,]\d+)*\.?", re.I)


def figuras(p: Pagina) -> None:
    """Un pie «ESQUEMA 5.1.» / «FIGURA 3» / «CUADRO 2» marca una figura: los renglones de encima que no son texto corrido
    (rótulos de cuerpo menor, cortos y sueltos) son la figura, no texto. El pie queda como pie, en su ortografía."""
    cuerpo = [l for l in p.lineas if not l.zona_clase]
    if len(cuerpo) < 5:
        return
    hmed = sorted(_alto(l) for l in cuerpo)[len(cuerpo) // 2]
    for k, l in enumerate(p.lineas):
        m = PIE.match(l.texto)
        if l.zona_clase or not m or len(l.texto) > 120:
            continue
        l.zona_clase = "Caption"
        l.texto = m.group(1).upper() + l.texto[m.end(1):] if len(l.texto.strip()) < 25 else l.texto
        cx = (l.caja[0] + l.caja[2]) / 2
        # la columna del pie: los renglones de cuerpo que lo cubren en horizontal
        col = [o for o in cuerpo if o.caja[0] <= cx <= o.caja[2] and _alto(o) >= 0.85 * hmed]
        if not col:
            continue
        izq, der = min(o.caja[0] for o in col), max(o.caja[2] for o in col)
        for o in p.lineas:
            if o is l or o.zona_clase or not (o.caja[3] <= l.caja[1] + 2 and izq - hmed <= o.caja[0] <= der):
                continue
            # entre el último renglón de texto corrido por encima y el pie
            encima = [q for q in col if q.caja[3] <= o.caja[1] + 1 and (q.caja[2] - q.caja[0]) > 0.6 * (der - izq)]
            techo = max((q.caja[3] for q in encima), default=0)
            menor = _alto(o) < 0.85 * hmed or (o.caja[2] - o.caja[0]) < 0.45 * (der - izq)
            if o.caja[1] >= techo - 1 and menor and not any(
                    q.caja[1] > o.caja[3] and q.caja[3] < l.caja[1] and (q.caja[2] - q.caja[0]) > 0.6 * (der - izq) and _alto(q) >= 0.85 * hmed
                    for q in col):
                o.zona_clase = "GraphicZone"


# ---------------------------------------------------------------- 4. romanos por contexto
ROMANO_FALSO = {"1": "I", "l": "I", "i": "I", "11": "II", "iI": "II", "Il": "II", "lI": "II", "ll": "II", "111": "III",
                "IIt": "III", "IIl": "III", "Ill": "III", "1I": "II", "I1": "II", "1V": "IV", "lV": "IV", "Vl": "VI", "V1": "VI"}
ROMANO = re.compile(r"^(?:X{0,3})(?:IX|IV|V?I{0,3})$")


def romanos(lineas: list[Linea]) -> None:
    """En una enumeración con romanos («I) … II) … III)»), los que el lector leyó como cifras o con minúsculas vuelven a
    ser romanos. Solo si en la misma serie hay al menos un romano de verdad de dos letras o más."""
    marcas = []
    for i, l in enumerate(lineas):
        m = re.match(r"^(\s*\(?)([IVXivxlt1]{1,5})(\)\s?)", l.texto)
        if m:
            marcas.append((i, m))
    if not any(ROMANO.match(m.group(2)) and len(m.group(2)) >= 2 for _, m in marcas):
        return
    for i, m in marcas:
        r = m.group(2)
        nuevo = r if ROMANO.match(r) else ROMANO_FALSO.get(r)
        if nuevo and nuevo != r:
            l = lineas[i]
            l.texto = m.group(1) + nuevo + l.texto[m.end(2):]
            if l.confs and len(r) != len(nuevo):
                l.confs = l.confs[: m.start(2)] + [l.conf] * len(nuevo) + l.confs[m.end(2):]


# ---------------------------------------------------------------- la página entera
def pulir_texto(p: Pagina) -> None:
    """Las correcciones de texto que valen para cualquier página, tenga maquetación o no: romanos y letras de las
    enumeraciones por su serie, y el punto suelto del número de un epígrafe («2.2 . TIPOS»)."""
    romanos([l for l in p.lineas if not es_fuera(l)])
    series_romanas(p.lineas)
    series_letras(p.lineas)
    series_arabigas(p.lineas)
    for l in p.lineas:
        if not es_fuera(l) and re.match(r"^\s*\d+(\.\d+)* \.", l.texto):
            i = l.texto.index(" .")
            l.texto = l.texto[:i] + l.texto[i + 1:]
            if l.confs and len(l.confs) > i:
                l.confs = l.confs[:i] + l.confs[i + 1:]


def pulir(p: Pagina, repetidas: set[str] | None = None) -> None:
    """Pule la ESTRUCTURA de una página sin maquetación (sin zonas): une filas partidas y clasifica cabeceras, folios,
    epígrafes, figuras y basura. Las páginas con zonas de una maquetación de verdad no se tocan (ya la tienen); las
    correcciones de texto van aparte, en pulir_texto, para todas."""
    if p.zonas or any(l.zona >= 0 or l.zona_clase for l in p.lineas) or not p.lineas:
        return
    repetidas = repetidas or set()
    vivas = [l for l in p.lineas if not es_basura(l, p.alto)]
    for l in p.lineas:
        if l not in vivas:
            l.zona_clase = "DigitizationArtefactZone"
    unidas = fusionar_filas(vivas)
    basura = [l for l in p.lineas if l.zona_clase]
    # las filas unidas sustituyen a sus trozos, en el mismo sitio del orden de lectura
    p.lineas = unidas + basura if basura else unidas
    cuerpo = [l for l in p.lineas if not l.zona_clase]
    if not cuerpo:
        return

    def columna(l):
        """Margen izquierdo y ancho de la columna del renglón (en una doble página partida hay dos)."""
        cx = (l.caja[0] + l.caja[2]) / 2
        vecinas = [o for o in cuerpo if o.caja[0] <= cx <= o.caja[2]] or cuerpo
        return (sorted(o.caja[0] for o in vecinas)[len(vecinas) // 4],
                sorted(o.caja[2] for o in vecinas)[(3 * len(vecinas)) // 4])

    for k, l in enumerate(p.lineas):
        if l.zona_clase:
            continue
        c = es_cabecera(l, p.alto, repetidas)
        if c:
            l.zona_clase = c
        elif es_titulo(l, p.lineas[k + 1] if k + 1 < len(p.lineas) else None, (c := columna(l))[0], c[1] - c[0],
                       p.lineas[k - 1] if k else None):
            l.zona_clase = "Section-header"
        elif es_versal_aislado(l, p.lineas[k - 1] if k else None, p.lineas[k + 1] if k + 1 < len(p.lineas) else None, p.alto):
            l.zona_clase = "Section-header"
        elif k and p.lineas[k - 1].zona_clase == "Section-header" and re.match(r"CAP[IÍ]TULO", p.lineas[k - 1].texto.strip(), re.I) \
                and len(l.texto) < 80 and sum(c.isupper() for c in l.texto) >= 0.6 * max(1, sum(c.isalpha() for c in l.texto)):
            l.zona_clase = "Section-header"  # el título del capítulo, debajo de «CAPÍTULO 5»
    figuras(p)


# ---------------------------------------------------------------- 5. series: enumeraciones y ejemplos numerados
def a_romano(n: int) -> str:
    out = ""
    for v, s in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out += s; n -= v
    return out


_CABEZA = re.compile(r"^(\s*)(\(?)([A-Za-z0-9]{0,5})(\)?)([.:]?)(?=\s|$|(?<=[.)])[A-ZÁÉÍÓÚ])")
_ROMANO_LEIDO = re.compile(r"^[IVXivxlt1mnr]{1,5}$")


def _parecido_romano(tok: str, n: int) -> bool:
    """¿Puede `tok` ser el romano `n` mal leído? «1»/«l»/«i» por I, «m»/«n»/«IIt»/«Il» por II o III, «rV» por IV."""
    r = a_romano(n)
    if tok == r or (tok == "" and r == "I"):
        return True
    if not _ROMANO_LEIDO.match(tok):
        return False
    t = tok.replace("1", "I").replace("l", "I").replace("i", "I").replace("t", "I").replace("r", "I").replace("v", "V").replace("x", "X")
    if t == r or ROMANO_FALSO.get(tok) == r:
        return True
    # «m» (tres palos) por III, «n» (dos) por II; un palo de más o de menos («I» por «II»); el «I» perdido de «I)»
    if (tok == "m" and r == "III") or (tok == "n" and r == "II") or (tok == "" and r == "I"):
        return True
    if set(t) <= {"I"} and set(r) <= {"I"} and abs(len(t) - len(r)) == 1:
        return True
    return len(t) == len(r) and sum(a != b for a, b in zip(t, r)) <= 1


def _series(lineas: list[Linea], es_item, valor, parecido, maximo: int = 8, corta=None, exacto=None):
    """Agrupa en series los renglones que empiezan por un numerador (`es_item`) con el mismo sangrado de la misma
    columna, sin más de 12 renglones entre uno y otro, y busca el comienzo que mejor cuadra. Devuelve
    [(serie [(índice, match)], comienzo)] de las series confirmadas: la mitad o más ya leída bien en su puesto."""
    items = []
    for i, l in enumerate(lineas):
        if es_fuera(l):
            continue
        m = _CABEZA.match(l.texto)
        if m and es_item(m):
            items.append((i, m))
    def cuadra(serie):
        """El comienzo con el que la serie cuadra (o None): todos sus numeradores parecidos al de su puesto, la mitad o
        más exactos y la mitad o más con su signo (un renglón que empieza por «I» o «a» sin más no hace serie)."""
        mejor = None
        for c in range(1, maximo + 1):
            buenos = sum((exacto or (lambda t, n: t == valor(n)))(m.group(3), k + c) for k, (_, m) in enumerate(serie))
            if all(parecido(m.group(3), k + c) for k, (_, m) in enumerate(serie)) and buenos and buenos * 2 >= len(serie):
                if mejor is None or buenos > mejor[1]:
                    mejor = (c, buenos)
        con_signo = sum(bool(m.group(4) or m.group(5)) for _, m in serie)
        return mejor[0] if mejor and con_signo * 2 >= len(serie) else None

    out, a = [], 0
    while a < len(items):
        i0, _ = items[a]
        x0, h0 = lineas[i0].caja[0], _alto(lineas[i0])
        serie, b, sig = [items[a]], a + 1, a + 1
        # crece mientras el siguiente numerador del mismo margen siga cuadrando
        while b < len(items) and items[b][0] - serie[-1][0] <= VENTANA_SERIE:
            if abs(lineas[items[b][0]].caja[0] - x0) < 0.8 * h0:
                if corta and corta(serie[-1][1].group(3), items[b][1].group(3)):
                    break
                if cuadra(serie + [items[b]]) is None:
                    # un numerador más adelante (del mismo margen) puede dar la razón a este
                    mas = next((it for it in items[b + 1:] if it[0] - items[b][0] <= VENTANA_SERIE
                                and abs(lineas[it[0]].caja[0] - x0) < 0.8 * h0), None)
                    if mas is None or cuadra(serie + [items[b], mas]) is None:
                        break
                serie.append(items[b])
                sig = b + 1
            b += 1
        if len(serie) >= 2:
            out.append((serie, cuadra(serie)))
            a = sig
        else:
            a += 1
    return out


def series_romanas(lineas: list[Linea]) -> None:
    """Listas «I) … II) … III)» en el mismo margen: cada numerador es el de su puesto en la serie («m)» entre «II)» y
    «IV)» es «III)»), aunque haya perdido el paréntesis («I hay implicatura» → «II) hay implicatura»)."""
    # (también «)» solo: el lector perdió el «I» de «I)»)
    es = lambda m: (bool(_ROMANO_LEIDO.match(m.group(3) or "-")) or (m.group(3) == "" and m.group(4) == ")")) and not m.group(5)
    def exacto(t, n):  # leído tal cual o con las confusiones seguras de palos (1, l, i, t, r por I)
        return t == a_romano(n) or t.replace("1", "I").replace("l", "I").replace("i", "I").replace("t", "I").replace("r", "I") == a_romano(n)

    for serie, c in _series(lineas, es, a_romano, _parecido_romano, exacto=exacto):
        # alguno tiene que estar escrito con letras: «1)», «1)» de una lista arábiga no es «I), II)»; «I) · I · IIr)», sí
        if not any(m.group(3) and not m.group(3).isdigit() for _, m in serie):
            continue
        for k, (i, m) in enumerate(serie):
            _cambiar_cabeza(lineas[i], m, a_romano(k + c), ")")


def series_arabigas(lineas: list[Linea]) -> None:
    """Listas «1) … 2) … 3)» o «1. … 2. …» en el mismo margen: el numerador que perdió su signo («1 «Haga usted…»
    entre «2)» y «3)») lo recupera. Solo cuadra con los números exactos en su puesto."""
    es = lambda m: m.group(3).isdigit() and len(m.group(3)) <= 2 and not m.group(2)
    for serie, c in _series(lineas, es, str, lambda t, n: t == str(n), maximo=1):
        con = [m for _, m in serie if m.group(4) or m.group(5)]
        if not con:
            continue
        signo = ")" if sum(bool(m.group(4)) for m in con) * 2 >= len(con) else "."
        for k, (i, m) in enumerate(serie):
            if not (m.group(4) or m.group(5)):
                _cambiar_cabeza(lineas[i], m, str(k + c), signo)


def series_letras(lineas: list[Linea]) -> None:
    """Listas «a) … b) … c)» o «a. … b. … c.»: la letra de su puesto, en minúscula, con el signo de la serie."""
    letra = lambda n: "abcdefghij"[n - 1] if 1 <= n <= 10 else "?"
    # (también «)» solo: el lector perdió la letra de «b)»)
    es = lambda m: ((len(m.group(3)) == 1 and m.group(3).isalpha()) or (m.group(3) == "" and m.group(4) == ")")) \
        and not (m.group(4) and m.group(5))
    # una serie de letras vuelve a empezar cuando la letra no avanza («a) b) … a) b) c)»)
    corta = lambda a, b: b != "" and a != "" and b.lower() <= a.lower()
    for serie, c in _series(lineas, es, letra, lambda t, n: t.lower() == letra(n) or t == "", maximo=6, corta=corta):
        parentesis = sum(m.group(4) == ")" for _, m in serie)
        signo = ")" if parentesis >= sum(m.group(5) == "." for _, m in serie) and parentesis else "."
        for k, (i, m) in enumerate(serie):
            _cambiar_cabeza(lineas[i], m, letra(k + c), signo)


def _cambiar_cabeza(l: Linea, m, nuevo: str, signo: str) -> None:
    """Pone `nuevo` + `signo` en lugar del numerador leído (con o sin paréntesis de cierre o punto)."""
    a, b = m.start(3), m.end(0)
    cola = l.texto[b:]
    t = l.texto[:a] + nuevo + signo + ("" if not cola or cola[0].isspace() else " ") + cola
    if t == l.texto:
        return
    if l.confs:
        l.confs = l.confs[:a] + [min(l.conf, 0.9)] * (len(t) - len(cola) - a) + l.confs[len(l.texto) - len(cola):]
    l.texto = t


_EJEMPLO = re.compile(r"^(\s*)\((\d{1,3})\)")
# un numerador de ejemplo roto: «(», «()», «(D)» o «)» solos; NO un romano ni una letra de subapartado («(ii)», «(b)»)
_EJEMPLO_ROTO = re.compile(r"^(\s*)(?:\((?=\s)|\(\)|\((?![IVXLivxl]{1,4}\)|[a-h]\))([^\s\d)(]{1,2})\)|\))(?=\s)")
# candidatos débiles: un romano «(II)» (¿un «(11)» mal leído?) o «(Juan» sin cierre en el renglón (¿«(7) Juan»?);
# solo cuentan si con ellos el hueco de la serie cuadra exactamente
_EJEMPLO_DEBIL = re.compile(r"^(\s*)(?:\(([IVXLivxl]{1,4})\)(?=\s)|\((?=[A-ZÁÉÍÓÚ][^)]*$))")
# un numerador con un paréntesis perdido: «4)» o «(8»
_EJEMPLO_COJO = re.compile(r"^(\s*)(?:(\d{1,3})\)|\((\d{1,3}))(?=\s)")


def _creciente_mas_larga(nums: list) -> set[int]:
    """Índices (de los que no son None) de la subsecuencia estrictamente creciente más larga."""
    idx = [k for k, n in enumerate(nums) if n is not None]
    mejor, ant = {}, {}
    for a, k in enumerate(idx):
        mejor[k], ant[k] = 1, None
        for j in idx[:a]:
            if nums[j] < nums[k] and mejor[j] + 1 > mejor[k]:
                mejor[k], ant[k] = mejor[j] + 1, j
    if not mejor:
        return set()
    k = max(mejor, key=mejor.get)
    out = set()
    while k is not None:
        out.add(k)
        k = ant[k]
    return out


def ejemplos_numerados(paginas: list[Pagina]) -> None:
    """Los ejemplos «(1) … (2) …» van correlativos en todo el documento: un numerador roto («(», «()», «(D)») entre
    dos buenos se rellena si los huecos cuadran (entre (1) y (4) caben dos rotos: (2) y (3))."""
    seq = []  # (página, línea, número o None, match)
    for pi, p in enumerate(paginas):
        for li, l in enumerate(p.lineas):
            if es_fuera(l):
                continue
            c = _EJEMPLO_COJO.match(l.texto)
            if c:  # «4)» / «(8»: el número está; el paréntesis se le devuelve si encaja en la serie (más abajo)
                seq.append((pi, li, int(c.group(2) or c.group(3)), c))
                continue
            m = _EJEMPLO.match(l.texto)
            if m:
                seq.append((pi, li, int(m.group(2)), m))
                continue
            m = _EJEMPLO_ROTO.match(l.texto)
            if m:
                seq.append((pi, li, None, m))
                continue
            m = _EJEMPLO_DEBIL.match(l.texto)
            if m:
                seq.append((pi, li, None, m))
    # los buenos que valen son los de la serie creciente más larga: una remisión a otro ejemplo al principio de un
    # renglón («(16) resume esta distinción») no es un ejemplo y no descuadra la serie
    validos = _creciente_mas_larga([s[2] for s in seq])
    seq = [s for k, s in enumerate(seq) if s[2] is None or k in validos]
    # el estilo del documento: si numera «3)» (un paréntesis), ese no es un «(3)» roto
    estilo_parentesis = sum(1 for s in seq if s[2] is not None and s[3].re is _EJEMPLO) > \
        sum(1 for s in seq if s[2] is not None and s[3].re is _EJEMPLO_COJO and s[3].group(2))
    for pi, li, n, m in seq:  # los cojos que encajan en la serie recuperan su paréntesis («4)» → «(4)»)
        if n is not None and m.re is _EJEMPLO_COJO and (estilo_parentesis or m.group(3)):
            l = paginas[pi].lineas[li]
            l.texto = m.group(1) + f"({n})" + l.texto[m.end():]
            l.confs = []
    buenos = [k for k, s in enumerate(seq) if s[2] is not None]
    # los rotos antes del primer bueno cuentan desde el (1) si cuadran: «(D)», «(», «(3)» → (1), (2), (3)
    if buenos and seq[buenos[0]][2] - 1 != buenos[0]:  # quitando los débiles de delante, ¿cuadra?
        debil = lambda s: s[3] is not None and s[3].re is _EJEMPLO_DEBIL
        fuertes = [s for s in seq[:buenos[0]] if not debil(s)]
        if seq[buenos[0]][2] - 1 == len(fuertes):
            seq = fuertes + seq[buenos[0]:]
            buenos = [k for k, s in enumerate(seq) if s[2] is not None]
    if buenos and seq[buenos[0]][2] - 1 == buenos[0] and buenos[0] > 0:
        seq.insert(0, (None, None, 0, None))
        buenos = [0] + [k + 1 for k in buenos]
    for a, b in zip(buenos, buenos[1:]):
        na, nb = seq[a][2], seq[b][2]
        todos = [k for k in range(a + 1, b) if seq[k][2] is None]
        fuertes = [k for k in todos if seq[k][3] is None or seq[k][3].re is not _EJEMPLO_DEBIL]
        rotos = todos if nb - na - 1 == len(todos) else fuertes
        if rotos and nb - na - 1 == len(rotos):
            for j, k in enumerate(rotos):
                pi, li, _, m = seq[k]
                l = paginas[pi].lineas[li]
                n = str(na + 1 + j)
                cola = l.texto[m.end():]
                l.texto = m.group(1) + f"({n})" + ("" if cola.startswith(" ") else " ") + cola
                if l.confs:
                    l.confs = [min(l.conf, 0.9)] * (len(l.texto) - len(cola)) + l.confs[-len(cola):] if cola else [min(l.conf, 0.9)] * len(l.texto)


_REF_PEGADA = re.compile(r"\((\d{1,3})([a-e])\b(?!\))")  # «(9a,» → «(9)a,»; «(6a)» es un ejemplo con subletra: se queda
_REF_VACIA = re.compile(r"\(\)")


def referencias(paginas: list[Pagina]) -> None:
    """Llamadas a ejemplos dentro del texto: el paréntesis de cierre perdido («(9a» → «(9)a») y la cifra perdida («en ()»
    → «en (6)»): la del ejemplo que viene en los renglones siguientes si la llamada lo anuncia («():»), si no la del
    último ejemplo."""
    ultimo, pos = None, []
    for pi, p in enumerate(paginas):
        for li, l in enumerate(p.lineas):
            if not es_fuera(l):
                pos.append((pi, li))
    numeros = {}
    for k, (pi, li) in enumerate(pos):
        m = _EJEMPLO.match(paginas[pi].lineas[li].texto)
        if m:
            numeros[k] = int(m.group(2))
    for k, (pi, li) in enumerate(pos):
        l = paginas[pi].lineas[li]
        if k in numeros:
            ultimo = numeros[k]
        t = _REF_PEGADA.sub(lambda m: f"({m.group(1)}){m.group(2)}", l.texto)
        if "()" in t:
            sig = next((numeros[j] for j in range(k + 1, min(k + 5, len(pos))) if j in numeros), None)

            def poner(m, t=t):
                anuncia = t[m.end():m.end() + 1] == ":"
                n = sig if (anuncia and sig) else ultimo
                return f"({n})" if n else m.group(0)
            t = _REF_VACIA.sub(poner, t)
        if t != l.texto:
            if l.confs and len(t) != len(l.texto):
                l.confs = []
            l.texto = t


# ---------------------------------------------------------------- 6. la numeración de los epígrafes
_NUM_EPIGRAFE = re.compile(r"^\s*((?:[\dlI]{1,2}(?:\s*\.\s*|\s+)){1,4}[\dlI]{0,2}\.?)(?=\s)")


def _siguientes(p: tuple) -> list[tuple]:
    """Los números que pueden venir detrás del epígrafe `p` (1.4.2 → 1.4.2.1, 1.4.3, 1.5, 2)."""
    out = [p + (1,)]
    for k in range(len(p), 0, -1):
        out.append(p[:k - 1] + (p[k - 1] + 1,))
    return out


def numeracion_epigrafes(paginas: list[Pagina]) -> None:
    """Los epígrafes numerados van en orden: un número roto entre dos buenos se rehace con el que le toca y cuyas
    cifras casan con las leídas («1.» entre 1.4.2 y 1.6 → 1.5; «18.» tras 1.7.1 → 1.8; «1.7. 1.» → 1.7.1)."""
    heads = []
    for p in paginas:
        for l in p.lineas:
            if es_titulo_zona(l):
                m = _NUM_EPIGRAFE.match(l.texto)
                if m:
                    crudo = m.group(1)
                    cifras = re.sub(r"[^\d]", "", crudo.replace("l", "1").replace("I", "1"))
                    partes = [x for x in re.split(r"\s*\.\s*|\s+", crudo.strip().rstrip(".")) if x]
                    num = tuple(int(x.replace("l", "1").replace("I", "1")) for x in partes) if all(
                        re.fullmatch(r"[\dlI]{1,2}", x) for x in partes) else None
                    heads.append([l, m, num, cifras])
    if len(heads) < 3:
        return
    # el número leído vale si sigue al anterior bueno
    def encaja(a, b):
        return a is None or b in _siguientes(a)

    ant = None
    for k, (l, m, num, cifras) in enumerate(heads):
        sig = next((h[2] for h in heads[k + 1:] if h[2] is not None), None)
        if num is not None and encaja(ant, num):
            bien = ".".join(str(x) for x in num) + "."
            if m.group(1).strip() != bien:  # «1.7. 1.» → «1.7.1.»
                l.texto = bien + l.texto[m.end(1):]
                l.confs = []
            ant = num
            continue
        if ant is None:
            continue
        # candidatos: lo que puede venir tras el anterior y antes del siguiente, con las cifras leídas en orden
        cands = [c for c in _siguientes(ant) if sig is None or sig in _siguientes(c) or sig == c]
        def casa(c):
            dig = "".join(str(x) for x in c)
            it = iter(dig)
            return all(ch in it for ch in cifras)  # las cifras leídas, subsecuencia de las del candidato
        buenos = [c for c in cands if casa(c)]
        if len(buenos) == 1:
            nuevo = ".".join(str(x) for x in buenos[0]) + "."
            l.texto = nuevo + l.texto[m.end(1):]
            l.confs = []
            ant = buenos[0]
        elif num is not None:
            ant = num


# ---------------------------------------------------------------- 7. documentos modernos en varias lenguas
# lo que el lector pone -> lo que suele haber debajo, si lo que pone no es una letra de las lenguas del documento
DIACRITICO_FALSO = {"ā": "ã", "ē": "ẽ", "ī": "ĩ", "ō": "õ", "ū": "ũ", "ȳ": "ỹ", "ă": "ã", "ĕ": "ẽ", "ĭ": "ĩ", "ŏ": "õ",
                    "ŭ": "ũ", "ë": "ê", "ö": "ô", "ä": "â", "ï": "î", "ü": "û", "è": "é", "ì": "í", "ò": "ó", "ù": "ú", "à": "á", "ſ": "f"}


def lenguas_del_documento(paginas: list[Pagina]) -> list[str]:
    from . import lexico

    ws = [w for p in paginas for w in lexico.palabras_de(p)]
    if lexico.es_antiguo(paginas):
        return []  # manuscritos e impresos antiguos: el macrón es abreviatura y la diéresis puede ser del original
    return lexico.lenguas_de(ws)


def diacriticos(paginas: list[Pagina], lenguas: list[str]) -> None:
    """Un diacrítico que no existe en ninguna lengua del documento es una lectura falsa de otro: en portugués, español o
    tupí, el macrón y la breve son la tilde nasal (Tupã → «Tupā»), la diéresis es la circunfleja (Aimbirê → «Aimbirë») y
    el grave del español es el agudo. En letón el macrón es una letra, en alemán la diéresis: ahí no se toca. La ü se
    deja en güe/güi si la lengua la tiene."""
    from . import lexico

    if not lenguas:
        return
    alf = lexico.alfabeto_de(lenguas)
    cambio = {a: b for a, b in DIACRITICO_FALSO.items() if a not in alf and (b in alf or b in "ĩẽũỹõãîû")}
    # las mayúsculas, si son una letra propia que tampoco existe en estas lenguas (la de «ſ» es «S»: no)
    cambio.update({a.upper(): b.upper() for a, b in list(cambio.items())
                   if a.upper() != a and a.upper().lower() == a and a.upper().lower() not in alf and a.upper() not in "SFIA"})
    if not cambio:
        return
    tabla = str.maketrans(cambio)
    for p in paginas:
        for l in p.lineas:
            t = l.texto
            t = re.sub(r"(?<=[gqGQ])ü(?=[eiéíEIÉÍ])", "\x00", t).translate(tabla).replace("\x00", "ü")
            if "·" in t and "·" not in alf:  # «merecer·algún»: el punto medio entre letras es un espacio (salvo l·l)
                t = re.sub(r"(?<=[^\W\d_])·(?=[^\W\d_])", lambda m: m.group(0) if t[m.start() - 1:m.start() + 2].lower() == "l·l" else " ", t)
            if t != l.texto:
                l.texto = t  # misma longitud: las confianzas siguen valiendo


# la cifra pegada detrás de dos letras o de un signo de cierre («bebedeira2», «1996)3.», «cuáles?9», «flores’8;»)
_LLAMADA = re.compile(r"(?:(?<=[^\W\d_]{2})|(?<=[)\]’'”\"?!»]))(\d{1,3})(?=[.,;:!?…»”\"')]*(?:\s|$))")


def llamadas_de_nota(paginas: list[Pagina]) -> None:
    """Las llamadas de nota (superíndices) que el lector pega a la palabra («bebedeira2», «chorá-lo6») van correlativas
    en el documento: si las cifras pegadas forman una serie creciente, son llamadas y se marcan «^n»."""
    vistas = []
    for pi, p in enumerate(paginas):
        for li, l in enumerate(p.lineas):
            if es_fuera(l):
                continue
            for m in _LLAMADA.finditer(l.texto):
                vistas.append((pi, li, m.start(), int(m.group(1))))
    if len(vistas) < 3:
        return
    crecen = sum(b[3] > a[3] for a, b in zip(vistas, vistas[1:]))
    if crecen < 0.6 * (len(vistas) - 1):
        return
    for pi, li, ini, n in sorted(vistas, key=lambda v: -v[2]):  # de derecha a izquierda: los índices no se mueven
        l = paginas[pi].lineas[li]
        l.texto = l.texto[:ini] + "^" + l.texto[ini:]
        if l.confs:
            l.confs = l.confs[:ini] + [1.0] + l.confs[ini:]
            if l.xs:
                l.xs = l.xs[:ini] + [l.xs[min(ini, len(l.xs) - 1)]] + l.xs[ini:]


_VERSO = re.compile(r"^(\s*)([0-9BSOolIZ]{1,4})([.,]) ")
_CIFRA = str.maketrans("BSOolIZ", "8500112")


_CONFUSION_CIFRAS = {"3": "8", "8": "3", "1": "7", "7": "1", "5": "6", "6": "5", "0": "8", "4": "9", "9": "4"}


def numeros_de_verso(p: Pagina) -> None:
    """Los números de verso al margen crecen con los renglones: «79,» es «79.», «B4,» es «84.» y una cifra confundida
    (3↔8, 1↔7…) se corrige solo si la lectura queda fuera del hueco entre sus vecinos y la confusión la devuelve al
    número que dan los renglones contados desde el anterior."""
    items = []
    for i, l in enumerate(p.lineas):
        if es_fuera(l):
            continue
        m = _VERSO.match(l.texto)
        if m:
            try:
                items.append([i, m, int(m.group(2).translate(_CIFRA))])
            except ValueError:
                pass
    if len(items) < 3:
        return
    for k, (i, m, v) in enumerate(items):
        ant = items[k - 1] if k else None
        sig = items[k + 1] if k + 1 < len(items) else None
        nuevo = v
        fuera = (ant and v <= ant[2]) or (sig and v >= sig[2])
        if fuera and ant:
            estimado = ant[2] + (i - ant[0])  # un renglón, un verso (aproximado: las acotaciones no cuentan)
            d = str(v)
            cands = {int(d[:j] + c + d[j + 1:]) for j, x in enumerate(d) for c in _CONFUSION_CIFRAS.get(x, "")}
            cands = [c for c in cands if c > ant[2] and (not sig or c < sig[2])]
            if cands:
                mejor = min(cands, key=lambda c: abs(c - estimado))
                if abs(mejor - estimado) <= 2:
                    nuevo = mejor
        texto = f"{nuevo}."
        if m.group(2) + m.group(3) != texto:
            l = p.lineas[i]
            a, b = m.start(2), m.end(3)
            l.texto = l.texto[:a] + texto + l.texto[b:]
            if l.confs and len(texto) != b - a:
                l.confs = []
        items[k][2] = nuevo


def distancia(a: str, b: str) -> int:
    """Distancia de edición (Levenshtein)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def personajes(paginas: list[Pagina]) -> None:
    """Los nombres de personaje al margen («Aimbirê:», «Guaixará:») se repiten: el más frecuente del documento corrige a
    sus parecidos («Afmbiré)» → «Aimbirê:»)."""
    pat = re.compile(r"^(\s*)([A-ZÁÉÍÓÚÂÊÎÔÛÃÕ][^\W\d_]{2,}(?:[ -][^\W\d_]{2,})?)([:;)=]|\s[=–-])")
    cuenta: Counter = Counter()
    for p in paginas:
        for l in p.lineas:
            m = pat.match(l.texto)
            if m and not es_fuera(l):
                cuenta[m.group(2)] += 1
    canon = [w for w, n in cuenta.most_common() if n >= 3]
    if not canon:
        return
    for p in paginas:
        for l in p.lineas:
            m = pat.match(l.texto)
            if not m or m.group(2) in canon:
                continue
            w = m.group(2)
            mejor = min(canon, key=lambda c: distancia(c.lower(), w.lower()))
            if distancia(mejor.lower(), w.lower()) <= 0.3 * max(len(w), len(mejor)) and cuenta[mejor] >= 3 * cuenta[w]:
                signo = ":" if m.group(3) in ";)" else m.group(3)
                l.texto = l.texto[:m.start(2)] + mejor + signo + l.texto[m.end(3):]
                l.confs = []
