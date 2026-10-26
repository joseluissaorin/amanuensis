"""Corrección léxica guiada por la CTC: la palabra que el lector vio y que además existe.

Para cada palabra dudosa (fuera del léxico y con alguna letra poco segura) se buscan candidatas del léxico: las que el
propio lector propone en su búsqueda en haz sobre los fotogramas de la palabra y las que están a una edición. Cada
candidata se puntúa con la verosimilitud CTC de las log-probabilidades del lector EN ESE TRAMO (no con un modelo de
lengua que no ha visto la página) y entra solo si el lector la ve casi tan bien como lo que leyó. Así «concento» pasa a
«concepto» (la p y la n se parecen en la CTC) pero «implicaturas», que el lector lee con seguridad, no se toca aunque
no esté en el léxico.

El léxico es doble: el general (formas con frecuencia) y el del propio documento, las palabras que el lector leyó con
total seguridad en otras partes del mismo texto, que además protegen el vocabulario técnico.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np

# letra = alfabética o signo combinante (las vocales del devanagari, los harakat del árabe, los diacríticos sueltos)
L_ = r"(?:[^\W\d_]|[\u0300-\u036f\u0483-\u0489\u064b-\u065f\u0670\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962\u0963])"
PALABRA = re.compile(rf"{L_}+(?:-{L_}+)*")
DUDA = 0.92  # una palabra es dudosa si alguna de sus letras baja de aquí
UMBRAL = -4.0  # se acepta si (log P CTC candidata - log P CTC lectura) + PESO_LEX·ln(1 + frecuencia) >= UMBRAL
PESO_LEX = 0.5
CAMBIO_LARGO = 4.0
ESPANOL = 0.95  # de las palabras leídas con seguridad, las que tienen que estar en el léxico (español moderno)
TILDE_FALTA = 300  # una forma acentuada rara se descarta si su hermana sin esa tilde es este tanto más frecuente
RARA = 20  # una forma del léxico más rara que esto no protege (la Wikipedia trae «concento» y otras faltas)
SEGURA = 0.97  # letras así de seguras hacen una palabra del autoléxico


def ruta_lexico() -> str | None:
    cache = os.path.join(os.environ.get("AMANUENSIS_CACHE") or os.path.expanduser("~/.cache/amanuensis"), "lexico_es.txt")
    for r in (os.environ.get("AMANUENSIS_LEXICO"), os.path.join(os.path.dirname(__file__), "lexico_es.txt"), cache):
        if r and os.path.exists(r):
            return r
    return None


@lru_cache(maxsize=2)
def cargar(ruta: str, minimo: int = 2) -> dict[str, int]:
    """Léxico {forma: frecuencia}. Higiene: una forma sin tilde cuya hermana con tilde es 20 veces más frecuente no
    cuenta («codigo», «teoria», «habia» son faltas de los subtítulos de donde sale la lista, no palabras)."""
    out = {}
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            p = linea.split()
            if len(p) >= 2 and p[1].isdigit():
                if int(p[1]) < minimo:
                    break
                out[p[0]] = int(p[1])
            elif len(p) == 1:
                out[p[0]] = 1
    mejor: dict[str, int] = {}
    for w, n in out.items():
        k = sin_tildes(w)
        if n > mejor.get(k, 0):
            mejor[k] = n
    # también las tildes mal puestas raras («linguístico» frente a «lingüístico»); las dos formas frecuentes de un par
    # legítimo («el»/«él», «que»/«qué») se quedan
    # se descarta: la forma sin tildes con hermana acentuada 20 veces más frecuente («codigo»); la acentuada rara cuya
    # hermana también acentuada lo es («linguístico» / «lingüístico»); y la acentuada cuya hermana sin esa tilde es
    # TILDE_FALTA veces más frecuente («varías» / «varias», 822). «cabrá» / «cabra» (51 veces) son dos palabras
    def descartar(w, n):
        k = sin_tildes(w)
        m = mejor.get(k, 0)
        if m < 20 * n:
            return False
        if k == w:
            return True
        hermana = max((f for f in hermanas.get(k, ()) if f != w), key=lambda f: out[f], default=None)
        return n < 1000 and (m >= TILDE_FALTA * n or (hermana is not None and sin_tildes(hermana) != hermana))

    hermanas: dict[str, list[str]] = {}
    for w in out:
        hermanas.setdefault(sin_tildes(w), []).append(w)
    fuera = {w: n for w, n in out.items() if descartar(w, n)}
    _DESCARTADAS[ruta] = {w: mejor[sin_tildes(w)] / max(1, n) for w, n in fuera.items()}
    return {w: n for w, n in out.items() if w not in fuera}


_DESCARTADAS: dict[str, dict[str, float]] = {}  # por léxico: forma descartada por la higiene -> cuántas veces más
# frecuente es su hermana («siguíente» → «siguiente», miles de veces)


def descartada(ruta: str | None, w: str) -> float:
    """Si la higiene descartó `w` en ese léxico, cuántas veces más frecuente es su hermana; si no, 0."""
    return _DESCARTADAS.get(ruta, {}).get(w, 0.0) if ruta else 0.0


@lru_cache(maxsize=2)
def por_clave(ruta: str) -> dict[str, list[str]]:
    """Formas del léxico por su clave sin tildes (para devolver las tildes perdidas): «codigo» → [«código»]."""
    out: dict[str, list[str]] = {}
    for w in cargar(ruta):
        out.setdefault(sin_tildes(w), []).append(w)
    return out


def ruta_lengua(lengua: str) -> str | None:
    r = os.path.join(carpeta_lexicos(), f"{lengua}.txt")
    if os.path.exists(r):
        return r
    return ruta_lexico() if lengua == "es" else None


@lru_cache(maxsize=40)
def lexico_de(lengua: str) -> dict[str, int]:
    r = ruta_lengua(lengua)
    return cargar(r) if r else {}


@lru_cache(maxsize=40)
def clave_de(lengua: str) -> dict[str, list[str]]:
    r = ruta_lengua(lengua)
    return por_clave(r) if r else {}


@lru_cache(maxsize=1)
def _perfiles() -> dict:
    """Perfiles de las 31 lenguas empaquetados (perfiles.json): la detección funciona sin léxicos ni red."""
    import json

    r = os.path.join(os.path.dirname(__file__), "perfiles.json")
    return json.load(open(r, encoding="utf-8")) if os.path.exists(r) else {}


@lru_cache(maxsize=40)
def perfil(lengua: str) -> tuple[frozenset, frozenset]:
    """(palabras funcionales, alfabeto) de una lengua. Funcionales: las 120 formas más frecuentes de su léxico.
    Alfabeto: las letras que pesan al menos dos cienmilésimas del texto (así la ü del español, «lingüística», entra).
    Vienen empaquetados en perfiles.json; si una lengua no está, se sacan de la cabeza de su léxico."""
    emp = _perfiles().get(lengua)
    if emp:
        return frozenset(emp["funcionales"]), frozenset(emp["alfabeto"])
    return perfil_de_fichero(lengua)


def perfil_de_fichero(lengua: str) -> tuple[frozenset, frozenset]:
    r = ruta_lengua(lengua)
    if not r:
        return frozenset(), frozenset()
    funcion, letras, total = [], Counter(), 0
    with open(r, encoding="utf-8") as f:
        for k, linea in enumerate(f):
            if k >= 30000:
                break
            p = linea.split()
            if len(p) < 2 or not p[1].isdigit():
                continue
            w, n = p[0], int(p[1])
            if len(funcion) < 120:
                funcion.append(w)
            for c in w:
                letras[c] += n
            total += n * len(w)
    return frozenset(funcion), frozenset(c for c, n in letras.items() if n >= 2e-5 * total and re.fullmatch(L_, c))


def escribir_perfiles(lenguas: list[str] | None = None) -> str:
    """Regenera perfiles.json (junto a este fichero) desde los léxicos de la caché."""
    import json

    out = {}
    for l in lenguas or LENGUAS:
        f, a = perfil_de_fichero(l)
        if f:
            out[l] = {"funcionales": sorted(f), "alfabeto": "".join(sorted(a))}
    r = os.path.join(os.path.dirname(__file__), "perfiles.json")
    json.dump(out, open(r, "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=True)
    _perfiles.cache_clear()
    perfil.cache_clear()
    return r


def lenguas_disponibles() -> list[str]:
    """Las lenguas que se pueden detectar (todas las de LENGUAS con perfil, salvo las que no separan palabras)."""
    return [l for l in LENGUAS if l not in SIN_ESPACIOS and perfil(l)[0]]


def hay_lexico(lengua: str) -> bool:
    return ruta_lengua(lengua) is not None


def preparar(lenguas: list[str], construir_faltan: bool = True, avisar=None) -> list[str]:
    """Deja listos los léxicos de `lenguas` (los construye si faltan y se puede) y devuelve los que hay."""
    hay = []
    for l in lenguas:
        if not hay_lexico(l) and construir_faltan:
            try:
                if avisar:
                    avisar(f"  construyo el léxico de «{l}» (solo la primera vez; baja las listas de frecuencias)")
                construir(l)
                lexico_de.cache_clear()
                clave_de.cache_clear()
            except Exception as e:  # sin red o fuente caída: la corrección sigue sin esa lengua
                if avisar:
                    avisar(f"  no he podido construir el léxico de «{l}» ({e}); sigo sin él")
        if hay_lexico(l):
            hay.append(l)
    return hay


def puntuar_lenguas(palabras: list[str], lenguas: list[str] | None = None) -> dict[str, float]:
    """Cuota de cada lengua en `palabras` por sus funcionales; una funcional que comparten varias lenguas («de», «que»,
    «a») reparte su peso entre ellas, así que «com», «não» o «los» deciden entre el portugués y el español."""
    lenguas = lenguas or lenguas_disponibles()
    perf = {l: perfil(l)[0] for l in lenguas}
    cuantas = Counter(w for f in perf.values() for w in f)
    n = max(1, len(palabras))
    return {l: sum(1 / cuantas[w] for w in palabras if w in f) / n for l, f in perf.items()}


def lenguas_de(palabras: list[str], minimo: float = 0.12, propias: float = 0.03) -> list[str]:
    """Las lenguas de un texto: la que más puntúa (si llega a `minimo`) y las que, además, ponen funcionales SUYAS
    (que no lo son de la principal) en al menos un `propias` de las palabras: un texto portugués con versos en español
    lleva las dos; uno español no arrastra al italiano por compartir «la» y «de»."""
    if len(palabras) < 15:
        return []
    pt = puntuar_lenguas(palabras)
    if not pt:
        return []
    primera = max(pt, key=pt.get)
    if pt[primera] < minimo:
        return []
    fp = perfil(primera)[0]
    out = [primera]
    for l in sorted(pt, key=lambda l: -pt[l]):
        if l == primera or pt[l] < pt[primera] / 4:
            continue
        suyas = perfil(l)[0] - fp
        if sum(w in suyas for w in palabras) >= propias * len(palabras):
            out.append(l)
    return out


def alfabeto_de(lenguas: list[str]) -> set[str]:
    out = set()
    for l in lenguas:
        out |= perfil(l)[1]
    return out


# ---------------------------------------------------------------- CTC
def ctc_logp(lp: np.ndarray, ids: list[int]) -> float:
    """log P(ids | lp) con el algoritmo hacia delante de la CTC (blanco = 0). lp: [T, C] log-probabilidades."""
    T = lp.shape[0]
    if not ids:
        return float(lp[:, 0].sum())
    ext = [0]
    for i in ids:
        ext += [i, 0]
    S = len(ext)
    if T < len(ids):
        return -math.inf
    a = np.full(S, -np.inf)
    a[0] = lp[0, 0]
    a[1] = lp[0, ext[1]]
    for t in range(1, T):
        prev = a
        a = np.full(S, -np.inf)
        a[0] = prev[0] + lp[t, 0]
        for s in range(1, S):
            v = np.logaddexp(prev[s], prev[s - 1])
            if s > 1 and ext[s] != 0 and ext[s] != ext[s - 2]:
                v = np.logaddexp(v, prev[s - 2])
            a[s] = v + lp[t, ext[s]]
    return float(np.logaddexp(a[-1], a[-2]))


def haz(lp: np.ndarray, chars: list[str], ancho: int = 16) -> list[tuple[str, float]]:
    """Búsqueda en haz por prefijos de la CTC sobre un tramo: las mejores lecturas con su log-probabilidad."""
    top = np.argsort(-lp, axis=1)[:, :5]
    beams = {"": (0.0, -np.inf)}  # prefijo -> (log p acabando en blanco, log p acabando en símbolo)
    for t in range(lp.shape[0]):
        nuevos: dict[str, list[float]] = {}

        def sumar(k, b, nb):
            if k not in nuevos:
                nuevos[k] = [-np.inf, -np.inf]
            nuevos[k][0] = np.logaddexp(nuevos[k][0], b)
            nuevos[k][1] = np.logaddexp(nuevos[k][1], nb)

        for pre, (pb, pnb) in beams.items():
            tot = np.logaddexp(pb, pnb)
            sumar(pre, tot + lp[t, 0], -np.inf)
            ult = pre[-1] if pre else None
            for c in top[t]:
                if c == 0:
                    continue
                ch = chars[c - 1]
                v = lp[t, c]
                if ch == ult:
                    sumar(pre, -np.inf, pnb + v)  # repetición sin blanco: el mismo símbolo
                    sumar(pre + ch, -np.inf, pb + v)  # tras un blanco: símbolo nuevo
                else:
                    sumar(pre + ch, -np.inf, tot + v)
        beams = dict(sorted(((k, (b, nb)) for k, (b, nb) in nuevos.items()), key=lambda x: -np.logaddexp(*x[1]))[:ancho])
    return [(k, float(np.logaddexp(*v))) for k, v in beams.items()]


# ---------------------------------------------------------------- candidatas
LETRAS = "abcdefghijklmnopqrstuvwxyzáéíóúüñ"


# lo que el lector lee -> lo que puede haber debajo (confusiones de forma típicas del OCR de imprenta)
CONFUSIONES = {"u": ["ti", "tí", "ii", "ri", "li"], "n": ["ri", "rí", "ti", "ii", "il"], "m": ["rn", "ni", "in", "ri", "rri", "iu", "nı"],
               "h": ["li", "lí", "b"], "d": ["cl"], "l": ["i", "t", "f", "í"], "i": ["l", "t", "í", "j"], "c": ["e"], "e": ["c", "é"],
               "rn": ["m"], "cl": ["d"], "li": ["h"], "ri": ["n"], "ii": ["ü", "u"], "o": ["ó", "e"], "a": ["á"], "j": ["i"],
               "f": ["t"], "t": ["f", "l"], "s": ["a"], "g": ["q", "o"], "q": ["g"]}


def confusiones_reglas(w: str, veces: int = 2) -> dict[str, frozenset]:
    """Formas a una o dos confusiones de OCR de `w` («prácucas» → «prácticas», «necesano» → «necesario»), cada una con
    las reglas (leído, debajo) que la producen."""
    out: dict[str, frozenset] = {}
    frente = {w: frozenset()}
    for _ in range(veces):
        nuevo: dict[str, frozenset] = {}
        for x, reglas in frente.items():
            for i in range(len(x)):
                for k, vs in CONFUSIONES.items():
                    if x.startswith(k, i):
                        for v in vs:
                            y = x[:i] + v + x[i + len(k):]
                            if y not in out and y not in nuevo:
                                nuevo[y] = reglas | {(k, v)}
        out.update(nuevo)
        frente = nuevo if len(nuevo) < 4000 else {}
    return out


def confusiones(w: str, veces: int = 2) -> set[str]:
    return set(confusiones_reglas(w, veces))


def confusiones_del_documento(palabras: list[str], lexico: dict[str, int], minimo: int = 6, frecuente: int = 200,
                              raices: int = 5) -> set:
    """Las confusiones SISTEMÁTICAS de este documento: las que convierten al menos `minimo` palabras distintas fuera del
    léxico en palabras frecuentes. Cada fuente tiene las suyas (una «ti» que el lector ve como «u», una «ri» como «n»): lo
    que se repite en muchas palabras distintas no es azar sino la letra de este libro."""
    n: dict = {}
    for w in set(palabras):
        for c, reglas in confusiones_reglas(w, 1).items():
            if lexico.get(c, 0) >= frecuente:
                for r in reglas:
                    n.setdefault(r, set()).add(w)
    unicas = set(palabras)
    # y que explique una buena parte de las palabras desconocidas que tienen esa letra (con letras comunes como la «a»,
    # alguna cae en el léxico por azar)
    # y de palabras de verdad distintas: «proferencia» y «proferencias» son una (en Grice, «o→e» salía de dos formas de
    # la misma palabra, un trozo y una palabra rara, y destrozaba el término técnico)
    return {r for r, ws in n.items() if len(ws) >= minimo and len({w[:6] for w in ws}) >= raices
            and len(ws) >= 0.05 * sum(r[0] in w for w in unicas)}


def una_edicion(w: str) -> set[str]:
    partes = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    borrar = {a + b[1:] for a, b in partes if b}
    trasponer = {a + b[1] + b[0] + b[2:] for a, b in partes if len(b) > 1}
    cambiar = {a + c + b[1:] for a, b in partes if b for c in LETRAS}
    poner = {a + c + b for a, b in partes for c in LETRAS}
    return borrar | trasponer | cambiar | poner


_SIN_TILDES = str.maketrans("áéíóúüàèìòùâêîôûäëïöñçÁÉÍÓÚÜÀÈÌÒÙÂÊÎÔÛÄËÏÖÑÇ", "aeiouuaeiouaeiouaeionc" + "AEIOUUAEIOUAEIOUAEIONC")


def sin_tildes(w: str) -> str:
    t = w.translate(_SIN_TILDES)
    if t.isascii():
        return t
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def con_caja(cand: str, modelo: str) -> str:
    """La candidata con las mayúsculas de la palabra leída."""
    if modelo.isupper() and len(modelo) > 1:
        return cand.upper()
    if modelo[:1].isupper():
        return cand[:1].upper() + cand[1:]
    return cand


# ---------------------------------------------------------------- el documento
FUNCION_ES = set("de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o este "
                 "es son entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante uno ni "
                 "contra ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos "
                 "mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros".split())
FUNCION_EN = set("the of and to in is that for it as with was on be by this are or from at an which have not but".split())


def palabras_de(p) -> list[str]:
    return [m.group(0).lower() for l in p.lineas for m in re.finditer(rf"{L_}+", l.texto)]


ANTIGUO = set("ſꝑꝓꝗꝙꝯ")


def es_antiguo(paginas) -> bool:
    """Impreso antiguo o manuscrito: las letras antiguas (ſ, ꝑ…) son frecuentes, una cada doscientas palabras o más. Una
    ſ suelta en un libro moderno es una «f» mal leída («inſormación»)."""
    n = sum(c in ANTIGUO for p in paginas for l in p.lineas for c in l.texto)
    palabras = sum(len(palabras_de(p)) for p in paginas)
    return n > 0 and n * 200 >= max(1, palabras)


def pagina_moderna(p, lenguas: list[str], lex) -> bool:
    """¿Se puede corregir esta página con los léxicos de hoy? De las palabras que el lector leyó con seguridad y que
    caben en el alfabeto de sus lenguas, el 95 % o más están en el léxico (un incunable se queda en el 83 %: «donzeles»,
    «dixiese», «verguença»), y sin letras antiguas (ſ, ꝑ)."""
    if not lenguas or es_antiguo([p]):
        return False
    alf = alfabeto_de(lenguas)
    seguras = buenas = 0
    ant = ""
    for l in p.lineas:
        t = l.texto.rstrip()
        cola_de_arriba = ant.endswith(("-", "¬"))
        ant = t
        if not (l.confs and len(l.confs) == len(l.texto)):
            continue
        ms = list(re.finditer(r"[^\W\d_]+", l.texto))
        for j, m in enumerate(ms):
            w = m.group(0).lower()
            # los trozos de una palabra partida a final de renglón no son palabras («inferen-», «-cionales»)
            if (j == len(ms) - 1 and l.texto[m.end():].strip()[:1] in ("-", "¬")) or (j == 0 and cola_de_arriba):
                continue
            if len(w) >= 3 and set(w) <= alf and min(l.confs[m.start():m.end()]) >= 0.95:
                seguras += 1
                buenas += lex(w) > 0
    return seguras >= 20 and buenas >= ESPANOL * seguras


def es_espanol_moderno(paginas, lexico: dict[str, int]) -> bool:
    """Compatibilidad: ¿el documento es español moderno?"""
    ws = [w for p in paginas for w in palabras_de(p)]
    return "es" in lenguas_de(ws)[:1] and all(pagina_moderna(p, ["es"], lambda w: lexico.get(w, 0)) for p in paginas if palabras_de(p))


def autolexico(paginas) -> Counter:
    n = Counter()
    for p in paginas:
        for l in p.lineas:
            for m in PALABRA.finditer(l.texto):
                cs = l.confs[m.start():m.end()] if l.confs and len(l.confs) == len(l.texto) else []
                if cs and min(cs) >= SEGURA:
                    n[m.group(0).lower()] += 1
    # una sola vez no protege: un error seguro de sí mismo («concento») también sale con letras seguras
    return Counter({k: v for k, v in n.items() if v >= 2})


class Lexicos:
    """Los léxicos de las lenguas de una página, vistos como uno: frecuencia = la mayor entre ellas."""

    def __init__(self, lenguas: list[str], fijo: dict[str, int] | None = None):
        self.lenguas = lenguas
        self.dicts = [fijo] if fijo is not None else [lexico_de(l) for l in lenguas]
        self.rutas = [] if fijo is not None else [ruta_lengua(l) for l in lenguas]
        self.claves = [] if fijo is not None else [clave_de(l) for l in lenguas]
        self.alfabeto = alfabeto_de(lenguas) if fijo is None else set()

    def __call__(self, w: str) -> int:
        return max((d.get(w, 0) for d in self.dicts), default=0)

    def __contains__(self, w: str) -> bool:
        return any(w in d for d in self.dicts)

    def falta_conocida(self, w: str) -> float:
        """Cuántas veces más frecuente es la hermana de `w` si `w` es una falta conocida (descartada) de estas lenguas."""
        return max((descartada(r, w) for r in self.rutas), default=0.0)

    def con_tildes(self, clave: str) -> list[str]:
        return [f for c in self.claves for f in c.get(clave, ())]

    def cabe(self, w: str) -> bool:
        """¿Está escrita con las letras de estas lenguas? («îandé» no cabe en el portugués)."""
        return not self.alfabeto or set(w.lower()) <= self.alfabeto


NADA = Lexicos([], {})


def corregir(paginas, lexico: dict[str, int] | None = None, registro: list | None = None,
             construir_faltan: bool = True, avisar=None) -> int:
    """Corrige en sitio las palabras dudosas de los renglones con log-probabilidades guardadas (`Linea._ctc`).

    Cada página con los léxicos de SUS lenguas (detectadas por las palabras funcionales; las 31 de LENGUAS si están en
    la caché, `python -m amanuensis.lexico construir`). Las páginas que no pasan por modernas (incunables, tupí sin
    léxico) y las palabras con letras ajenas a esas lenguas se corrigen solo con el autoléxico: el vocabulario que el
    propio documento lee con seguridad."""
    propio = autolexico(paginas)
    doc = [w for p in paginas for w in palabras_de(p)]
    lenguas_doc = lenguas_de(doc)
    if lexico is None and lenguas_doc:
        lenguas_doc = preparar(lenguas_doc, construir_faltan, avisar) or []
    lexs = {}
    for k, p in enumerate(paginas):
        if lexico is not None:
            lx = Lexicos(["es"], lexico)
        else:
            ws = palabras_de(p)
            ls = lenguas_de(ws) if len(ws) >= 40 else []
            ls = [l for l in ls if l in lenguas_doc] or lenguas_doc[:1]
            lx = Lexicos(ls) if ls else NADA
        lexs[k] = lx if lx is not NADA and pagina_moderna(p, lx.lenguas or ["es"], lx) else NADA
    if registro is not None and lenguas_doc:
        registro.append(("(lenguas del documento)", " ".join(lenguas_doc), 0.0, 0.0))
    # los léxicos de las lenguas del documento protegen también en las páginas que no corrigen con ellos («foge»)
    lx_doc = Lexicos(lenguas_doc) if lenguas_doc and lexico is None else NADA

    # primera pasada: las palabras fuera del léxico de las páginas modernas, para aprender sus confusiones sistemáticas
    # (contra el léxico general, no el del documento: un error sistemático también se repite con seguridad)
    sistematicas = set()
    modernas = [k for k in lexs if lexs[k] is not NADA]
    if modernas:
        fuera, freq_ref = [], Counter()
        for k in modernas:
            lx = lexs[k]
            for l in paginas[k].lineas:
                if getattr(l, "_ctc", None) is None:
                    continue
                ms = list(PALABRA.finditer(l.texto))
                for j, m in enumerate(ms):
                    w = m.group(0).lower()
                    if j == len(ms) - 1 and l.texto[m.end():].strip()[:1] in ("-", "¬"):
                        continue  # trozo de palabra partida
                    if len(w) >= 3 and lx(w) < RARA and lx.cabe(w):
                        fuera.append(w)
                        for c in confusiones_reglas(w, 1):
                            freq_ref[c] = max(freq_ref[c], lx(c))
        sistematicas = confusiones_del_documento(fuera, freq_ref)
        for w in list(propio):
            if all(w not in lexs[k] for k in modernas) and any(
                    freq_ref.get(c, 0) >= 200 and r <= sistematicas for c, r in confusiones_reglas(w, 1).items()):
                del propio[w]  # el autoléxico no protege lo que una confusión sistemática hace palabra («comunicauvo»)
    if registro is not None and sistematicas:
        registro.append(("(confusiones de este documento)", " ".join(f"{a}→{b}" for a, b in sorted(sistematicas)), 0.0, 0.0))

    cache_lx: dict[tuple, Lexicos] = {}

    def lexicos_del_renglon(p, k, lx_pag):
        """La lengua del renglón por su entorno (dos renglones arriba y abajo): en el teatro de Anchieta las estrofas
        cambian de lengua dentro de la página. Si el entorno no decide, la de la página."""
        if lx_pag is NADA or len(lenguas_doc) < 2:
            return lx_pag
        ws = [m.group(0).lower() for o in p.lineas[max(0, k - 2):k + 3] for m in re.finditer(rf"{L_}+", o.texto)]
        ls = [l for l in lenguas_de(ws, minimo=0.08) if l in lenguas_doc][:1] if len(ws) >= 15 else []
        if not ls:
            return lx_pag
        clave_ls = tuple(ls)
        if clave_ls not in cache_lx:
            cache_lx[clave_ls] = Lexicos(list(ls))
        return cache_lx[clave_ls]

    cambios = 0
    for k_p, p in enumerate(paginas):
        lx_pagina = lexs[k_p]
        for k, l in enumerate(p.lineas):
            lx_pag = lexicos_del_renglon(p, k, lx_pagina)
            ctc = getattr(l, "_ctc", None)
            if ctc is None or not l.confs or len(l.confs) != len(l.texto):
                continue
            lp, fot, chars = ctc
            if len(fot) != len(l.texto):
                continue
            idx = {c: i + 1 for i, c in enumerate(chars)}
            sig = p.lineas[k + 1].texto if k + 1 < len(p.lineas) else ""
            resto = PALABRA.match(sig.lstrip())
            partida = l.texto.rstrip().endswith(("-", "¬")) and resto is not None
            ant = p.lineas[k - 1].texto.rstrip() if k else ""
            cola_de_arriba = ant.endswith(("-", "¬")) and len(ant) > 1 and ant[-2].isalpha()
            nuevo, desp = l.texto, 0
            ms = list(PALABRA.finditer(l.texto))
            for j, m in enumerate(ms):
                w = m.group(0)
                a, b = m.start(), m.end()
                cola = resto.group(0).lower() if partida and j == len(ms) - 1 and l.texto[b:].strip() in ("-", "¬") else ""
                # la palabra con letras ajenas a las lenguas de la página, solo con el vocabulario del documento
                lx = lx_pag if lx_pag.cabe(w + cola) else NADA
                lw = (w + cola).lower()
                # el autoléxico no protege una falta conocida: la forma sin tildes con hermana acentuada («codigo» /
                # «código»), o la acentuada que el léxico descartó porque su hermana es mil veces más frecuente
                # («siguíente» / «siguiente»); «tupã» frente a «tupa» no llega ahí y se protege
                if sin_tildes(lw) == lw:
                    hermanas = [f for f in lx.con_tildes(sin_tildes(lw)) if f != lw]
                else:
                    # descartada por la higiene, o ni siquiera en el léxico con una hermana sin tildes muy frecuente
                    hermana = max((lx(f) for f in lx.con_tildes(sin_tildes(lw)) if f != lw), default=0)
                    falta = lx.falta_conocida(lw) >= TILDE_FALTA or (lw not in lx and hermana >= 10000)
                    hermanas = ["(falta conocida)"] if falta else []
                existe = (lx(lw) >= RARA or (lw in lx and propio[lw] >= 1) or (propio[lw] >= 1 and not hermanas)
                          or (lx is NADA and lx_doc(lw) >= RARA))
                if len(w) < 3 or existe or (cola == "" and w[-1:] == "-") or (j == 0 and cola_de_arriba):
                    continue
                antes = l.texto[:a].rstrip()
                if w[:1].isupper() and antes and antes[-1] not in ".:!?¿¡«—(":
                    continue  # nombres propios a media frase
                f0 = fot[a - 1] + 1 if a > 0 else 0
                f1 = fot[b] if b < len(fot) else lp.shape[0]
                seg = lp[f0:f1]
                if seg.shape[0] < len(w) or any(ch not in idx for ch in w):
                    continue
                base = ctc_logp(seg, [idx[c] for c in w])

                def frecuencia(c):
                    return lx(c) + 1000 * propio[c]

                def vale(c):
                    return c + cola in lx or propio[c + cola] > 0

                cands = set()
                for t, _ in haz(seg, chars, 24):
                    t = t.strip().lower()
                    if t and vale(t):
                        cands.add(t)
                conf_reglas = confusiones_reglas(w.lower())
                conf_w = set(conf_reglas)
                de_confusion = set()
                for c in una_edicion(w.lower()) | conf_w | {w.lower()}:
                    if vale(c):
                        cands.add(c)
                    for f in lx.con_tildes(sin_tildes(c) + sin_tildes(cola)) if not cola else ():
                        cands.add(f)
                        if c in conf_w or c == w.lower():
                            de_confusion.add(f)
                    if c in conf_w:
                        de_confusion.add(c)
                for c in list(cands):
                    f = lx(c + cola)
                    if f < 500 and not propio[c + cola]:
                        vec = [v for v in una_edicion(c) if lx(v + cola) >= 100 * max(f, 1)]
                        if vec:
                            cands.discard(c)
                            cands.update(vec)
                mejor, mp, mv = None, -math.inf, -math.inf
                for c in cands:
                    cc = con_caja(c, w)
                    if any(ch not in idx for ch in cc):
                        continue
                    lc = ctc_logp(seg, [idx[ch] for ch in cc])
                    delta = lc - base
                    reglas = conf_reglas.get(c)
                    if reglas and reglas <= sistematicas and frecuencia(c + cola) >= 20:
                        delta = max(delta, -1.0)
                    v = delta + PESO_LEX * math.log1p(frecuencia(c + cola)) - (CAMBIO_LARGO if len(cc) != len(w) and c not in de_confusion else 0)
                    if v > mv:
                        mejor, mp, mv = cc, lc, v
                # solo con el autoléxico (tupí, páginas antiguas) la CTC tiene que estar de acuerdo, y nunca se cambian
                # solo tildes: en una lengua sin léxico, «nhẽ» y «nhe» pueden ser palabras distintas
                if mejor and lx is NADA and (mp - base < -2.0 or sin_tildes(mejor.lower()) == sin_tildes(w.lower())):
                    mejor = None
                if mejor and mejor != w and mv >= UMBRAL:
                    if registro is not None:
                        registro.append((w + ("-" + cola if cola else ""), mejor, round(mp - base, 2), round(mv, 2)))
                    nuevo = nuevo[:a + desp] + mejor + nuevo[b + desp:]
                    desp += len(mejor) - len(w)
                    cambios += 1
            if nuevo != l.texto:
                if len(nuevo) != len(l.texto):
                    l.confs = []
                    l.xs = []
                l.texto = nuevo
    return cambios


# ---------------------------------------------------------------- los 31 idiomas
LENGUAS = "ar bg cs da de el en es et fi fr hi hr hu id it ja ko lt lv nl pl pt ro ru sk sl sv tr uk vi".split()
# lenguas sin espacios entre palabras: la corrección por palabra no se aplica
SIN_ESPACIOS = {"ja"}
FUENTES_SUBTITULOS = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/{l}/{l}_full.txt"
FUENTES_WIKI = {"ar": "arwiki-2022-08-29", "de": "dewiki-2022-08-29", "en": "enwiki-2023-04-13", "es": "eswiki-2022-08-29",
                "fr": "frwiki-2022-08-29", "it": "itwiki-2022-08-30", "ja": "jawiki-2022-08-29", "nl": "nlwiki-2022-08-29",
                "pl": "plwiki-2022-08-29", "pt": "ptwiki-2022-08-29", "ru": "ruwiki-2022-08-29", "sv": "svwiki-2022-08-30",
                "uk": "ukwiki-2022-08-30", "vi": "viwiki-2022-08-30"}
URL_WIKI = "https://raw.githubusercontent.com/IlyaSemenov/wikipedia-word-frequency/master/results/{}.txt"
ESCALA = 1e9  # cada fuente se lleva a mil millones de palabras: las frecuencias valen igual en todas las lenguas
MAX_FORMAS = 800_000


def carpeta_lexicos() -> str:
    return os.path.join(os.environ.get("AMANUENSIS_CACHE") or os.path.expanduser("~/.cache/amanuensis"), "lexicos")


def construir(lengua: str, carpeta: str | None = None) -> str:
    """Baja las listas de frecuencias de `lengua` (subtítulos de OpenSubtitles 2018 y, si la hay, la Wikipedia), las
    lleva a la misma escala, las suma y escribe <carpeta>/<lengua>.txt («forma frecuencia», de más a menos)."""
    import urllib.request

    carpeta = carpeta or carpeta_lexicos()
    os.makedirs(carpeta, exist_ok=True)
    total: Counter = Counter()
    urls = [FUENTES_SUBTITULOS.format(l=lengua)] + ([URL_WIKI.format(FUENTES_WIKI[lengua])] if lengua in FUENTES_WIKI else [])
    for url in urls:
        n: Counter = Counter()
        req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "amanuensis-lexicos/1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            datos = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                import gzip

                datos = gzip.decompress(datos)
            for linea in datos.decode("utf-8", "replace").splitlines():
                p = linea.rsplit(" ", 1)
                if len(p) == 2 and p[1].isdigit() and re.fullmatch(rf"{L_}+(?:['’-]{L_}+)*", p[0]):
                    n[p[0].lower()] += int(p[1])
        s = sum(n.values()) or 1
        for w, c in n.items():
            total[w] += c * ESCALA / s
    ruta = os.path.join(carpeta, f"{lengua}.txt")
    tmp = f"{ruta}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for w, c in total.most_common(MAX_FORMAS):
            if c < 1:
                break
            f.write(f"{w} {int(round(c))}\n")
    os.replace(tmp, ruta)
    return ruta


if __name__ == "__main__":
    import sys
    from concurrent.futures import ThreadPoolExecutor

    if len(sys.argv) >= 2 and sys.argv[1] == "construir":
        pedidas = sys.argv[2:] or LENGUAS
        with ThreadPoolExecutor(6) as ex:
            for l, r in zip(pedidas, ex.map(construir, pedidas)):
                print(l, r, os.path.getsize(r) // 1024, "KB", flush=True)
