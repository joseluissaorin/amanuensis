"""Documento leído y sus salidas: txt, md, json, html, hocr, alto (v4), page (PAGE 2019), tei y spdf.

Estructura (con la maquetación del modelo completo): las líneas se agrupan en BLOQUES por su zona y en orden de
lectura (párrafos, títulos, listas, notas al margen ancladas a la línea del texto principal que tienen al lado,
notas al pie, cabeceras y números de página, tablas por celdas, fórmulas, figuras con su pie). Sin zonas (lector
Mínimo/Ligero o capa de texto del PDF), los bloques son párrafos, como en las versiones anteriores.

Letras dudosas: cada línea lleva la probabilidad de cada carácter (`confs`); las que quedan por debajo de
`Documento.umbral_duda` (0,5 por defecto) se marcan en TEI (<unclear cert>), hOCR (x_wconf y x_confs), ALTO (WC y CC),
PAGE (conf de la palabra) y JSON (confs y `dudas`).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from xml.sax.saxutils import escape as _escape, quoteattr as _quoteattr

_NO_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿\ud800-\udfff]")
UMBRAL_DUDA = 0.5
# bits de estilo tipográfico (orden de la cabeza tipográfica de «estructura», maqueta/est_cabezas.py BITS)
ESTILO = {"negrita": 1, "cursiva": 2, "versalita": 4, "subrayado": 8, "color": 16}
REND_TEI = {1: "bold", 2: "italic", 4: "smallcaps", 8: "underline", 16: "color(red)"}
ESTILO_ALTO = {1: "bold", 2: "italics", 4: "smallcaps", 8: "underline"}


def escape(t) -> str:  # quita los caracteres que XML 1.0 no admite (algunas capas de texto de PDF los traen)
    return _escape(_NO_XML.sub("", str(t)))


def quoteattr(t) -> str:
    return _quoteattr(_NO_XML.sub("", str(t)))


FORMATOS = ("md", "txt", "json", "html", "hocr", "alto", "page", "tei", "spdf")
EXTENSION = {"md": ".md", "txt": ".txt", "json": ".json", "html": ".html", "hocr": ".hocr", "alto": ".alto.xml",
             "page": ".page.xml", "tei": ".tei.xml", "spdf": ".spdf"}

# zona de la maquetación (SegmOnto y DocLayNet) -> papel del bloque
ROL = {"MainZone": "texto", "Text": "texto", "TitlePageZone": "texto", "DamageZone": "texto", "List-item": "lista",
       "MarginTextZone": "margen", "RunningTitleZone": "cabecera", "Page-header": "cabecera", "NumberingZone": "numero",
       "QuireMarksZone": "signatura", "Page-footer": "pie_pagina", "Title": "titulo", "Section-header": "titulo",
       "Footnote": "nota", "Caption": "pie", "TableZone": "tabla", "Table": "tabla", "DropCapitalZone": "capitular",
       "Formula": "formula", "MusicZone": "musica", "DigitizationArtefactZone": "artefacto",
       "StampZone": "figura", "SealZone": "figura", "GraphicZone": "figura", "Picture": "figura"}
# zona de imagen -> tipo de figura (figuras.py)
TIPO_FIGURA = {"Picture": "figura", "GraphicZone": "ilustracion", "DropCapitalZone": "capitular", "StampZone": "sello_tinta",
               "SealZone": "sello", "Formula": "formula"}
FW = {"cabecera": "header", "numero": "pageNum", "signatura": "sig", "pie_pagina": "footer"}


@dataclass
class Linea:
    caja: list  # x0, y0, x1, y1 en px de la página
    texto: str
    conf: float = 1.0
    xs: list = field(default_factory=list)  # x de cada carácter (px de la página), si se conoce
    poligono: list | None = None  # contorno de la línea (maquetación), [[x, y], ...]
    base: list | None = None  # línea base, [[x, y], ...]
    tipo: str | None = None  # tipo de línea SegmOnto (maquetación)
    zona_clase: str | None = None  # clase de la zona de la maqueta que la contiene
    en_figura: str | None = None  # id de la figura dentro de la cual está (rótulos, leyendas de un sello)
    pie_de: bool = False  # forma parte del pie de una figura
    confs: list = field(default_factory=list)  # probabilidad de cada carácter (misma longitud que texto), si se conoce
    zona: int = -1  # índice de su zona en Pagina.zonas
    # estilo tipográfico (cabeza tipográfica de la maquetación): palabras {palabra, ini, fin, negrita: p, cursiva: p, ...}
    # (el formato de estructura.py) o tramos [t0, t1, bits] en fracciones de la línea, con bits = suma de ESTILO
    estilos: list = field(default_factory=list)

    def bits_por_letra(self) -> list:
        """Estilo (bits) de cada carácter del texto, según la posición de su centro en la línea."""
        t = self.texto
        if not self.estilos or not t:
            return [0] * len(t)
        x0, x1 = self.caja[0], self.caja[2]
        if len(self.xs) == len(t) and x1 > x0:
            pos = [(x - x0) / (x1 - x0) for x in self.xs]
        else:
            pos = [(k + 0.5) / len(t) for k in range(len(t))]
        out = [0] * len(t)
        for e in self.estilos:
            if isinstance(e, dict):  # palabra de la cabeza tipográfica: {palabra, ini, fin, negrita: p, cursiva: p, ...}
                bits = sum(v for k, v in ESTILO.items() if e.get(k, 0) >= 0.5)
                for k in range(max(0, int(e.get("ini", 0))), min(len(t), int(e.get("fin", 0)))):
                    out[k] |= bits
            else:  # tramo [t0, t1, bits] en fracciones de la línea
                t0, t1, bits = e
                for k, u in enumerate(pos):
                    if t0 <= u <= t1:
                        out[k] |= int(bits)
        return out

    def letras(self, umbral: float = 0.5) -> list:
        """[(carácter, cert o None, bits)] del texto sin los espacios de los extremos."""
        t = self.texto
        cs = self.confs if len(self.confs) == len(t) else None
        bs = self.bits_por_letra()
        ini, fin = len(t) - len(t.lstrip()), len(t.rstrip())
        return [(t[k], cs[k] if cs and cs[k] < umbral and not t[k].isspace() else None, bs[k]) for k in range(ini, fin)]

    def palabras(self):
        """[(texto, caja, confs)] a partir de las posiciones de los caracteres (o repartiendo el ancho)."""
        x0, y0, x1, y1 = self.caja
        t = self.texto
        xs = self.xs if len(self.xs) == len(t) and t else None
        if xs is None:
            n = max(1, len(t))
            xs = [x0 + (i + 0.5) * (x1 - x0) / n for i in range(len(t))]
        cs = self.confs if len(self.confs) == len(t) else None
        paso = (x1 - x0) / max(1, len(t))
        out, i = [], 0
        while i < len(t):
            if t[i] == " ":
                i += 1
                continue
            j = i
            while j < len(t) and t[j] != " ":
                j += 1
            a, b = sorted((xs[i], xs[j - 1]))
            out.append((t[i:j], [max(x0, a - paso / 2), y0, min(x1, b + paso / 2), y1], cs[i:j] if cs else None))
            i = j
        return out

    def tramos(self, umbral: float = UMBRAL_DUDA):
        """[(texto, cert o None)]: tramos seguros y tramos dudosos (cert = la probabilidad mínima del tramo)."""
        t, cs = self.texto, self.confs
        if len(cs) != len(t) or not t:
            return [(t, None)] if t else []
        out = []
        for ch, c in zip(t, cs):
            duda = c < umbral and not ch.isspace()
            if out and (out[-1][1] is not None) == duda:
                out[-1] = (out[-1][0] + ch, min(out[-1][1], c) if duda else None)
            else:
                out.append((ch, c if duda else None))
        return out


@dataclass
class Pagina:
    ancho: int
    alto: int
    lineas: list
    fuente: str = "ocr"  # ocr | texto (capa de texto del PDF)
    modelo: str = ""
    segundos: float = 0.0
    zonas: list = field(default_factory=list)  # [{"clase", "tipo", "caja", "conf"}] (maquetación)
    figuras: list = field(default_factory=list)  # imágenes extraídas (figuras.py)
    tablas: list = field(default_factory=list)  # [{"caja", "filas", "columnas"}] (maquetación v4)
    estructura: str | None = None  # «diccionario» | «comic» | None (estructura.py)
    lector: str = ""  # lector que leyó la página (selector por escritura)

    @property
    def texto(self):
        return "\n".join(l.texto for l in self.lineas)

    def como_cadena(self) -> dict:
        """La página en el formato de la cadena de página entera (el que espera estructura.py)."""
        return {"lineas": [{"caja": l.caja, "texto": l.texto, "zona": l.zona, "tipo": l.tipo, "conf": l.conf} for l in self.lineas],
                "zonas": self.zonas, "tablas": self.tablas, "texto": self.texto, "estructura": self.estructura}


@dataclass
class Documento:
    nombre: str
    paginas: list
    imagenes: dict = field(default_factory=dict, repr=False)  # ruta relativa -> imagen PIL (recortes a guardar)
    umbral_duda: float = UMBRAL_DUDA
    # Markdown, TXT y HTML: por defecto, texto de lectura (sin folios, cabeceras ni marcas de página; párrafos y palabras
    # partidas unidos también de una página a otra). fiel=True: página a página, con el mobiliario en comentarios.
    fiel: bool = False

    def exportar(self, formato: str) -> str:
        return {"txt": txt, "md": md, "json": a_json, "html": html, "hocr": hocr, "alto": alto, "page": page, "tei": tei,
                "spdf": spdf}[formato](self)

    def guardar_imagenes(self, carpeta) -> int:
        from pathlib import Path

        for rel, im in self.imagenes.items():
            f = Path(carpeta) / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            im.save(f, "WEBP", quality=90, method=4)
        return len(self.imagenes)


# ---------------------------------------------------------------- bloques
def _corrido(l) -> bool:
    # los renglones dentro de una figura (rótulos, leyendas) siguen en el texto: la maquetación toma a veces por imagen
    # bloques de texto, y perder texto es peor que repetir un rótulo (decisión del 19-10-2026, ver INFORME-TECNICO)
    return bool(l.texto.strip()) and not l.pie_de


def _rol(p: Pagina, l) -> str:
    if 0 <= l.zona < len(p.zonas):
        z = p.zonas[l.zona]
        return ROL.get(z.get("tipo") or z.get("clase"), "texto")
    if l.zona_clase:
        return ROL.get(l.zona_clase, "texto")
    return "texto"


_ENUM = re.compile(r"^\s*(\(\d{1,3}\)|\d{1,3}\)|\(?[IVXivx]{1,5}\)|\(?[a-z]\))\s")


def _partir_parrafos(p: Pagina, idx: list) -> list:
    """Índices de líneas -> grupos de párrafo, por hueco vertical, cambio de cuerpo, sangría y fin de frase."""
    ls = [p.lineas[i] for i in idx]
    if not ls:
        return []
    hs = sorted(l.caja[3] - l.caja[1] for l in ls)
    hmed = hs[len(hs) // 2]
    izq = sorted(l.caja[0] for l in ls)[len(ls) // 4]
    grupos, act = [], [idx[0]]
    for k, (a, b) in enumerate(zip(ls, ls[1:])):
        hueco = b.caja[1] - a.caja[3]
        ha, hb = a.caja[3] - a.caja[1], b.caja[3] - b.caja[1]
        # sangría (más de un cuerpo) tras fin de frase, o un numerador al principio («(2) …», «II) …», «b) …»)
        nuevo = (abs(ha - hb) > 0.25 * max(ha, hb) or hueco > 0.8 * hmed or b.caja[1] < a.caja[1] - hmed
                 or (b.caja[0] - izq > 1.0 * hmed and a.texto.rstrip()[-1:] in ".:!?»\")")
                 or (_ENUM.match(b.texto) is not None and a.texto.rstrip()[-1:] not in "-¬"))
        # la palabra partida sigue en el renglón de abajo aunque el alto de la caja cambie (descendentes, página
        # enderezada): «narcisis-» / «mo realista» no abre párrafo
        ta, tb = a.texto.rstrip(), b.texto.lstrip()
        if (nuevo and len(ta) > 1 and ta[-1] in "-¬⸗" and ta[-2].isalpha() and tb[:1].islower()
                and hueco <= 0.8 * hmed and b.caja[1] >= a.caja[1] - hmed):
            nuevo = False
        if nuevo:
            grupos.append(act); act = []
        act.append(idx[k + 1])
    grupos.append(act)
    return grupos


def unir_letras(lineas, umbral: float = UMBRAL_DUDA) -> list:
    """Como unir(), pero letra a letra con su duda y su estilo: [(carácter, cert o None, bits)]."""
    act = []
    for l in lineas:
        ls = l.letras(umbral)
        if not ls:
            continue
        if act and act[-1][0] in "-¬⸗" and len(act) > 1 and act[-2][0].isalpha():
            act = act[:-1] + ls
        else:
            act = act + [(" ", None, 0)] + ls if act else ls
    return act


def tramos_ricos(letras) -> list:
    """[(texto, cert o None, bits)] agrupando letras seguidas con la misma duda y el mismo estilo (los espacios
    toman el estilo de la letra anterior, para que una palabra en cursiva seguida de otra no se corte)."""
    out = []
    for k, (ch, c, b) in enumerate(letras):
        if ch.isspace():  # el espacio va dentro del estilo solo si la letra siguiente lo continúa
            sig = next((x[2] for x in letras[k + 1:] if not x[0].isspace()), 0)
            b = out[-1][2] if out and out[-1][2] == sig else 0
        duda = c is not None
        if out and (out[-1][1] is not None) == duda and out[-1][2] == b:
            t, c0, _ = out[-1]
            out[-1] = (t + ch, min(c0, c) if duda else None, b)
        else:
            out.append((ch, c, b))
    return out


def _bits_rend(b: int) -> str:
    return " ".join(v for k, v in REND_TEI.items() if b & k)


def html_rico(lineas, umbral: float = UMBRAL_DUDA) -> str:
    """Texto corrido en HTML con estilos (<b>, <i>, versalita, <u>, color) y letras dudosas resaltadas."""
    o = []
    for t, c, b in tramos_ricos(unir_letras(lineas, umbral)):
        x = escape(t)
        if c is not None:
            x = f'<span class="duda" title="probabilidad {c:.2f}">{x}</span>'.replace(f"{c:.2f}", f"{c:.2f}".replace(".", ","))
        if b & 4:
            x = f'<span class="versalita">{x}</span>'
        if b & 16:
            x = f'<span class="color">{x}</span>'
        if b & 8:
            x = f"<u>{x}</u>"
        if b & 2:
            x = f"<i>{x}</i>"
        if b & 1:
            x = f"<b>{x}</b>"
        o.append(x)
    return "".join(o)


def md_rico(lineas) -> str:
    """Texto corrido en Markdown con negrita y cursiva (las demás marcas no tienen equivalente)."""
    o = []
    for t, _, b in tramos_ricos(unir_letras(lineas, 0.0)):
        m = ("**" if b & 1 else "") + ("*" if b & 2 else "")
        core = t.strip()
        if not m or not core:
            o.append(t)
            continue
        ini, fin = t[: len(t) - len(t.lstrip())], t[len(t.rstrip()):]
        o.append(f"{ini}{m}{core}{m[::-1]}{fin}")
    return "".join(o)


def _con_estilos(p) -> bool:
    return any(l.estilos for l in p.lineas)


def unir(lineas) -> str:
    """Texto corrido de varias líneas: quita los guiones de corte."""
    act = ""
    for l in lineas:
        t = l.texto.strip()
        if not t:
            continue
        if act[-1:] in "-¬⸗" and len(act) > 1 and act[-2].isalpha():
            act = act[:-1] + t
        else:
            act = (act + " " + t) if act else t
    return act


def parrafos(p: Pagina, con_indice: bool = False) -> list:
    """(Compatibilidad con 0.4) párrafos de texto corrido; con con_indice, (texto, índice de su primera línea)."""
    idx = [i for i, l in enumerate(p.lineas) if _corrido(l)]
    out = [(unir([p.lineas[i] for i in g]), g[0]) for g in _partir_parrafos(p, idx)]
    return out if con_indice else [t for t, _ in out]


def _celdas(p: Pagina, idx: list) -> list:
    """Líneas de una tabla -> filas de celdas [[ [índices de línea], ... ], ...] con la rejilla de la maqueta v4 si la hay
    (la tabla cuya caja más solapa con las líneas); si no, filas por solape vertical y celdas por orden en la fila."""
    ls = [p.lineas[i] for i in idx]
    x0, y0 = min(l.caja[0] for l in ls), min(l.caja[1] for l in ls)
    x1, y1 = max(l.caja[2] for l in ls), max(l.caja[3] for l in ls)

    def solape(c):
        return max(0, min(x1, c[2]) - max(x0, c[0])) * max(0, min(y1, c[3]) - max(y0, c[1]))

    t = max(p.tablas, key=lambda t: solape(t["caja"]), default=None)
    if t is not None and solape(t["caja"]) > 0 and len(t.get("filas", [])) >= 1 and len(t.get("columnas", [])) >= 1:
        F, C = t["filas"], t["columnas"]

        def cerca(v, ints):
            return min(range(len(ints)), key=lambda k: 0 if ints[k][0] <= v <= ints[k][1] else min(abs(v - ints[k][0]), abs(v - ints[k][1])))

        rej = [[[] for _ in C] for _ in F]
        for i, l in zip(idx, ls):
            rej[cerca((l.caja[1] + l.caja[3]) / 2, F)][cerca((l.caja[0] + l.caja[2]) / 2, C)].append(i)
        for fila in rej:
            for c in fila:
                c.sort(key=lambda i: (p.lineas[i].caja[1], p.lineas[i].caja[0]))
        return [f for f in rej if any(f)]
    filas = []
    for i in sorted(idx, key=lambda i: (p.lineas[i].caja[1] + p.lineas[i].caja[3]) / 2):
        l = p.lineas[i]
        cy = (l.caja[1] + l.caja[3]) / 2
        if filas:
            u = p.lineas[filas[-1][0]]
            if u.caja[1] <= cy <= u.caja[3]:
                filas[-1].append(i)
                continue
        filas.append([i])
    return [[[i] for i in sorted(f, key=lambda i: p.lineas[i].caja[0])] for f in filas]


MARGEN_FW = 0.12  # franja de arriba y de abajo de la página (fracción del alto) donde viven folios y cabeceras


def es_folio(t: str) -> bool:
    """¿Número de página? «12», «01 / 19», «- 7 -», «xiv»."""
    s = "".join(c for c in t if not c.isspace() and c not in "/-·.–—")
    return 0 < len(s) <= 8 and (s.isdigit() or all(c in "ivxlcdmIVXLCDM" for c in s))


def es_letras_esparcidas(t: str) -> bool:
    """Rótulo con letra espaciada que el lector parte en fichas («L A C O M U N I C A C I Ó N», «SUB GÉ NE RO»): corto,
    sin minúsculas y con la mayoría de sus fichas de una o dos letras. Un título de verdad con palabras cortas («1.1. EN
    BUSCA DE REGULARIDADES») y los romanos («CAPÍTULO IV») no lo son."""
    if len(t) > 60 or any(c.islower() for c in t):
        return False
    fichas = [f for f in t.split() if any(c.isalpha() for c in f)]
    if len(fichas) < 2:
        return False
    sueltas = sum(len(f) == 1 for f in fichas)
    if len(fichas) >= 4 and sueltas * 2 > len(fichas):
        return True
    cortas = sum(len(f) <= 2 and not all(c in "IVXLCDM" for c in f) for f in fichas)
    return cortas * 5 >= len(fichas) * 3


def _mobiliario(p: Pagina, b) -> str | None:
    """Sin maquetación, los folios y las cabeceras corrientes (arriba o abajo de la página) pasan a «numero» y
    «cabecera» (<fw> en TEI, comentario en Markdown) y los restos de una sola letra o cifra a «artefacto»: siguen en las
    salidas, pero fuera del texto corrido."""
    t = b["texto"].strip()
    if len(b["lineas"]) == 1 and sum(c.isalnum() for c in t) <= 1 and (p.lineas[b["lineas"][0]].conf < 0.6 or not re.fullmatch(r"\(?\w[).]?\)?", t)):
        return "artefacto"
    if not p.alto:
        return None
    ys = [p.lineas[i].caja for i in b["lineas"]]
    borde = all(c[3] <= p.alto * MARGEN_FW for c in ys) or all(c[1] >= p.alto * (1 - MARGEN_FW) for c in ys)
    if not borde or len(t) > 80:
        return None
    if es_folio(t):
        return "numero"
    if es_letras_esparcidas(t):
        return "cabecera"
    return None


def _mobiliario_fuera(p: Pagina, out: list) -> list:
    """El mobiliario sale del flujo (el de arriba delante, el de abajo detrás) y el párrafo que partía se vuelve a unir:
    el orden de lectura mete a veces el folio entre el primer renglón de la página y el resto."""
    fw = [b for b in out if b["rol"] != "texto"]
    if not fw:
        return out
    texto, partido = [], False
    for b in out:
        if b["rol"] != "texto":
            partido = True
            continue
        if partido and texto and texto[-1]["texto"] and b["texto"][:1].islower() and texto[-1]["texto"][-1] not in ".!?:»”\"":
            a = texto[-1]
            a["lineas"] = a["lineas"] + b["lineas"]
            a["texto"] = unir([p.lineas[i] for i in a["lineas"]])
        else:
            texto.append(b)
        partido = False
    abajo = [b for b in fw if p.lineas[b["lineas"][0]].caja[1] >= p.alto / 2]
    return [b for b in fw if b not in abajo] + texto + abajo


def bloques(p: Pagina) -> list:
    """Bloques de la página en orden de lectura: dicts {"rol", "lineas" (índices), "texto", ["celdas"], ["ancla"],
    ["nivel"]}. Las notas al margen llevan «ancla»: índice de la línea del texto principal junto a la que van."""
    idx = [i for i, l in enumerate(p.lineas) if _corrido(l)]
    if not idx:
        return []
    con_zonas = any(l.zona >= 0 or l.zona_clase for l in p.lineas)
    if not con_zonas:
        out = [{"rol": "texto", "lineas": g, "texto": unir([p.lineas[i] for i in g])} for g in _partir_parrafos(p, idx)]
        for b in out:
            b["rol"] = _mobiliario(p, b) or b["rol"]
        return _mobiliario_fuera(p, out)
    # tramos seguidos de la misma zona (en orden de lectura)
    tramos = []
    for i in idx:
        l = p.lineas[i]
        clave = (l.zona if l.zona >= 0 else l.zona_clase, _rol(p, l))
        if tramos and tramos[-1][0] == clave:
            tramos[-1][1].append(i)
        else:
            tramos.append((clave, [i]))
    # una tabla es un solo bloque aunque el orden la parta
    out, tablas_vistas = [], {}
    for (z, rol), ls in tramos:
        if rol == "tabla":
            if z in tablas_vistas:
                tablas_vistas[z]["lineas"] += ls
                continue
            b = {"rol": "tabla", "lineas": list(ls)}
            tablas_vistas[z] = b
            out.append(b)
        elif rol in ("texto", "lista", "nota"):
            for g in _partir_parrafos(p, ls):
                out.append({"rol": rol, "lineas": g})
        elif rol == "titulo":  # cada epígrafe numerado (o «CAPÍTULO N») empieza bloque; si no, es el renglón de abajo
            g = []
            for i in ls:
                if g and (re.match(r"^\s*(\d{1,2}(\.\d{1,2})*\.\s|cap[ií]tulo\b)", p.lineas[i].texto, re.I)
                          or re.match(r"^\s*cap[ií]tulo\b", p.lineas[g[-1]].texto, re.I)):
                    out.append({"rol": "titulo", "lineas": g}); g = []
                g.append(i)
            out.append({"rol": "titulo", "lineas": g})
        else:
            out.append({"rol": rol, "lineas": ls})
    for b in out:
        b["texto"] = unir([p.lineas[i] for i in b["lineas"]])
        if b["rol"] == "tabla":
            b["celdas"] = _celdas(p, b["lineas"])
    # capitular: su letra va al principio del bloque siguiente de texto
    for k, b in enumerate(out):
        if b["rol"] == "capitular" and k + 1 < len(out) and out[k + 1]["rol"] in ("texto", "titulo"):
            sig = out[k + 1]
            sig["capitular"] = b["texto"]
            sig["texto"] = b["texto"] + sig["texto"]
            sig["lineas"] = b["lineas"] + sig["lineas"]
            b["absorbido"] = True
    out = [b for b in out if not b.get("absorbido")]
    # notas al margen: se anclan a la línea de texto principal más cercana en vertical y van detrás del bloque de esa línea
    principales = [i for b in out if b["rol"] in ("texto", "lista", "titulo") for i in b["lineas"]]
    if principales:
        margen = [b for b in out if b["rol"] == "margen"]
        resto = [b for b in out if b["rol"] != "margen"]
        for b in margen:
            y = p.lineas[b["lineas"][0]].caja[1]
            b["ancla"] = min(principales, key=lambda i: abs(p.lineas[i].caja[1] - y))
        donde = {i: k for k, b in enumerate(resto) for i in b["lineas"]}
        colas = {}
        for b in margen:
            colas.setdefault(donde[b["ancla"]], []).append(b)
        out = []
        for k, b in enumerate(resto):
            out.append(b)
            out += sorted(colas.get(k, []), key=lambda m: p.lineas[m["lineas"][0]].caja[1])
    return out


def secuencia(p: Pagina, capitulares: bool = False):
    """Bloques y figuras intercalados en orden: [("b", bloque) | ("f", figura)]."""
    figs = [f for f in p.figuras if capitulares or f["tipo"] != "capitular"]
    anclas = sorted(((ancla(p, f), n, f) for n, f in enumerate(figs)), key=lambda t: (t[0], t[1]))
    out, k = [], 0
    pos = {i: n for n, i in enumerate(range(len(p.lineas)))}
    for b in bloques(p):
        ini = pos[b["lineas"][0]]
        while k < len(anclas) and anclas[k][0] <= ini:
            out.append(("f", anclas[k][2])); k += 1
        out.append(("b", b))
    out += [("f", a[2]) for a in anclas[k:]]
    return out


def ancla(p: Pagina, fig) -> int:
    """Índice de la línea (en orden de lectura) delante de la cual va la figura: la primera línea de texto
    corrido que empieza por debajo del borde superior de la figura y la solapa en horizontal; si no hay, al final."""
    c = fig["caja"]
    for i, l in enumerate(p.lineas):
        if _corrido(l) and l.caja[1] >= c[1] and min(l.caja[2], c[2]) - max(l.caja[0], c[0]) > 0:
            return i
    return len(p.lineas)


def _estructura_rica(p: Pagina):
    if not p.estructura:
        return None
    from . import estructura

    return estructura


# ---------------------------------------------------------------- texto y Markdown
def txt(d: Documento) -> str:
    if d.fiel:
        return "\n\f\n".join("\n".join(l.texto for l in p.lineas) for p in d.paginas) + "\n"
    out = []
    for rol, t in corrido(d):
        out.append(re.sub(r"^#+\s*", "", t) if rol == "titulo" else t.replace("  \n", "\n") if rol == "verso" else t)
    return "\n\n".join(out) + "\n"


def es_verso(p: Pagina, idx: list, lleno: float) -> bool:
    """¿Estrofa? Dos renglones o más, casi todos cortos (menos del 70 % de un renglón lleno de la página) y ninguno
    partido con guion: la prosa llena el renglón hasta el margen; el verso, no."""
    ls = [p.lineas[j] for j in idx]
    if len(ls) < 2 or not lleno:
        return False
    # referencia absoluta: la hoja (media página si es una doble página partida); en una página toda de verso, el
    # renglón «lleno» de la propia página es un verso
    hoja = p.ancho / 2 if "doble página" in (p.fuente or "") else p.ancho
    tope = max(0.7 * lleno / 0.9, 0.55 * hoja) if hoja else 0.7 * lleno / 0.9
    anchos = [l.caja[2] - l.caja[0] for l in ls]
    cortos = sum(a < tope for a in anchos)
    # y desiguales: la prosa justificada (también la de una columna estrecha) tiene los renglones del mismo ancho
    cuerpo = anchos[:-1]
    media = sum(cuerpo) / len(cuerpo)
    desiguales = (max(cuerpo) - min(cuerpo)) > 0.08 * media if len(cuerpo) >= 2 else anchos[0] < 0.6 * tope
    return cortos >= 0.8 * len(ls) and desiguales and not any(l.texto.rstrip()[-1:] in "-¬" for l in ls[:-1])


FIN_DE_FRASE = ".!?:;»”\")…]"
FUERA_DE_LECTURA = set(FW) | {"musica", "artefacto", "figura"}


def corrido(d: Documento, rico: bool = True) -> list:
    """El documento como texto de lectura: [(rol, texto en Markdown)] sin mobiliario (folios, cabeceras, signaturas,
    restos del escaneo, rótulos de figura) y con lo que el cambio de página corta vuelto a unir: la palabra partida
    («comu-» / «nicación») y el párrafo que sigue en la página siguiente (sin fin de frase y empezando en minúscula). Las
    notas al pie y al margen van al final, en el orden en que aparecen."""
    out, notas = [], []
    sigue = False  # ¿el último párrafo emitido acaba en un renglón lleno (o en guion), es decir, continúa?
    for i, p in enumerate(d.paginas):
        er = _estructura_rica(p)
        if er is not None:
            out.append(("texto", er.a_markdown(p.como_cadena(), p.estructura)))
            sigue = False
            continue
        con_estilo = rico and _con_estilos(p)
        n = 0
        anchos = sorted(l.caja[2] - l.caja[0] for l in p.lineas if not l.zona_clase and len(l.texto) > 20)
        lleno = anchos[len(anchos) * 3 // 4] * 0.9 if anchos else 0
        for tipo, x in secuencia(p):
            if tipo == "f":
                alt = x["alt"].replace("[", "(").replace("]", ")")
                out.append(("figura", f"![{alt}]({x['fichero']})" + (f"\n\n*{x['pie']}*" if x["pie"] else "")))
                continue
            rol, t = x["rol"], x["texto"]
            if rol in FUERA_DE_LECTURA:
                continue
            # una letra o cifra suelta que no es un numerador («a)», «3.») es un resto del escaneo
            if rol == "texto" and sum(c.isalnum() for c in t) <= 1 and not re.fullmatch(r"\(?\w[).]\)?", t.strip()):
                continue
            if con_estilo and rol in ("texto", "lista", "margen", "nota"):
                t = md_rico([p.lineas[j] for j in x["lineas"]])
            ultima = p.lineas[x["lineas"][-1]] if x.get("lineas") else None
            llena = bool(ultima and lleno and ultima.caja[2] - ultima.caja[0] >= lleno)
            if rol == "texto" and es_verso(p, x.get("lineas", []), lleno):
                # verso: un renglón, un verso (salto de línea de Markdown); nunca se une con lo que viene
                versos = [p.lineas[j].texto.strip() for j in x["lineas"] if p.lineas[j].texto.strip()]
                out.append(("verso", "  \n".join(versos)))
                sigue = False
                continue
            if rol == "texto":
                if sigue and out and out[-1][0] == "texto" and t[:1].isalpha():
                    a = out[-1][1].rstrip()
                    if a[-1:] in "-¬⸗" and len(a) > 1 and a[-2].isalpha():  # «comu-» / «nicación»
                        out[-1] = ("texto", a[:-1] + t.lstrip())
                        sigue = llena or t.rstrip()[-1:] in "-¬⸗"
                        continue
                    if a[-1:] not in FIN_DE_FRASE and t[:1].islower():  # el párrafo sigue en la otra página u hoja
                        out[-1] = ("texto", a + " " + t.lstrip())
                        sigue = llena or t.rstrip()[-1:] in "-¬⸗"
                        continue
                out.append(("texto", t))
                sigue = llena or t.rstrip()[-1:] in "-¬⸗"
            elif rol == "titulo":
                out.append(("titulo", titulo_md(t)))
                sigue = False
            elif rol == "lista":
                out.append(("lista", "- " + t))
                sigue = False
            elif rol == "margen":
                n += 1
                ref = f"[^m{i + 1}-{n}]"
                if out:
                    out[-1] = (out[-1][0], out[-1][1] + ref)
                notas.append(f"{ref}: {t} *(al margen)*")
            elif rol == "nota":
                notas.append(t)
            elif rol == "tabla":
                out.append(("tabla", _md_tabla(p, x)))
            elif rol == "pie":
                out.append(("pie", f"*{t}*"))
            elif rol == "formula":
                out.append(("formula", f"`{t}`"))
            else:
                out.append(("texto", t))
    if notas:
        out.append(("notas", "---"))
        out += [("nota", t) for t in notas]
    return out


def _md_tabla(p: Pagina, b) -> str:
    filas = [[unir([p.lineas[i] for i in c]).replace("|", "\\|") for c in f] for f in b["celdas"]]
    n = max(len(f) for f in filas)
    filas = [f + [""] * (n - len(f)) for f in filas]
    s = ["| " + " | ".join(filas[0]) + " |", "|" + "---|" * n]
    s += ["| " + " | ".join(f) + " |" for f in filas[1:]]
    return "\n".join(s)


def titulo_md(t: str) -> str:
    """Epígrafe en Markdown: el nivel por la numeración («3.» ##, «3.1.» ###) y las versalitas, que el lector lee como
    mayúsculas o mezcladas («TIPOS DE CONTENIDOS iMPlÍCITOS»), en minúscula con la inicial en mayúscula."""
    c = re.match(r"^\s*cap[ií]tulo\s+(\S+)\s*$", t, re.I)
    if c:
        return "# CAPÍTULO " + c.group(1)
    m = re.match(r"^\s*(\d{1,2}(?:\.\d{1,2})*\.?)\s+(.*)$", t)
    num, resto = (m.group(1), m.group(2)) if m else ("", t.strip())
    letras = [c for c in resto if c.isalpha()]
    mayus = sum(c.isupper() for c in letras)
    raro = re.search(r"\w[a-zà-ÿ][A-ZÀ-Þ]|\b[a-zà-ÿ][A-ZÀ-Þ]{2,}", resto)
    if letras and (mayus >= 0.6 * len(letras) or (num and raro)) and not re.match(r"CAP[IÍ]TULO", resto, re.I):
        resto = re.sub(r"^([¿¡«“‘(]*)(\w)", lambda m: m.group(1) + m.group(2).upper(), resto.lower(), count=1)
    nivel = 2 + num.rstrip(".").count(".") if num else 2
    return "#" * nivel + " " + (num + " " if num else "") + resto


def md(d: Documento) -> str:
    if not d.fiel:
        return "\n\n".join(t for _, t in corrido(d)) + "\n"
    partes = []
    for i, p in enumerate(d.paginas):
        if len(d.paginas) > 1:
            partes.append(f"<!-- página {i + 1} -->")
        er = _estructura_rica(p)
        if er is not None:
            partes.append(er.a_markdown(p.como_cadena(), p.estructura))
            continue
        notas, n = [], 0
        rico = _con_estilos(p)
        for tipo, x in secuencia(p):
            if tipo == "f":
                alt = x["alt"].replace("[", "(").replace("]", ")")
                partes.append(f"![{alt}]({x['fichero']})" + (f"\n\n*{x['pie']}*" if x["pie"] else ""))
                continue
            rol, t = x["rol"], x["texto"]
            if rico and rol in ("texto", "lista", "margen", "nota"):
                t = md_rico([p.lineas[j] for j in x["lineas"]])
            if rol == "texto":
                partes.append(t)
            elif rol == "titulo":
                partes.append(titulo_md(t))
            elif rol == "lista":
                partes.append("- " + t)
            elif rol == "margen":
                n += 1
                ref = f"[^m{i + 1}-{n}]"
                partes[-1] = partes[-1] + ref if partes else ref
                notas.append(f"{ref}: {t} *(al margen)*")
            elif rol == "nota":
                notas.append(t)
            elif rol == "tabla":
                partes.append(_md_tabla(p, x))
            elif rol == "pie":
                partes.append(f"*{t}*")
            elif rol == "formula":
                partes.append(f"`{t}`")
            elif rol in FW or rol in ("musica", "artefacto", "figura"):
                partes.append(f"<!-- {rol.replace('_', ' ')}: {t.replace('--', '- -')} -->")
            else:
                partes.append(t)
        if notas:
            partes.append("---")
            partes += notas
    return "\n\n".join(partes) + "\n"


def html(d: Documento) -> str:
    s = ['<!doctype html>', '<html lang="und"><head><meta charset="utf-8">', f"<title>{escape(d.nombre)}</title>",
         '<meta name="generator" content="amanuensis">',
         "<style>body{max-width:44rem;margin:2rem auto;font:1.05rem/1.55 Georgia,serif;padding:0 1rem}figure{margin:1.5rem 0}"
         "figure img{max-width:100%}figcaption,.nota,.fw{font-size:.9rem;color:#555}aside.margen{float:right;clear:right;width:11rem;"
         "margin:0 -12rem .5rem 1rem;font-size:.85rem;color:#6b4a35}table{border-collapse:collapse;margin:1rem 0}"
         "td{border:1px solid #ccc;padding:.2rem .5rem;vertical-align:top}.duda{background:#f4d9c8;cursor:help}"
         ".versalita{font-variant:small-caps}.color{color:#a3271f}</style></head><body>"]
    if not d.fiel:
        return "\n".join(s + [_html_corrido(d), "</body></html>"]) + "\n"
    for i, p in enumerate(d.paginas):
        s.append(f'<section class="pagina" id="p{i + 1}">')
        for tipo, x in secuencia(p):
            if tipo == "f":
                cap = f"<figcaption>{escape(x['pie'])}</figcaption>" if x["pie"] else ""
                s.append(f'<figure id="{x["id"]}" class="{x["tipo"]}"><img src="{escape(x["fichero"])}" alt={quoteattr(x["alt"])} loading="lazy">{cap}</figure>')
                continue
            rol, t = x["rol"], html_rico([p.lineas[j] for j in x["lineas"]], d.umbral_duda)
            if rol == "titulo":
                s.append(f"<h2>{t}</h2>")
            elif rol == "lista":
                s.append(f"<ul><li>{t}</li></ul>")
            elif rol == "margen":
                s.append(f'<aside class="margen">{t}</aside>')
            elif rol == "nota":
                s.append(f'<p class="nota">{t}</p>')
            elif rol == "tabla":
                filas = "".join("<tr>" + "".join(f"<td>{html_rico([p.lineas[j] for j in c], d.umbral_duda)}</td>" for c in f) + "</tr>" for f in x["celdas"])
                s.append(f"<table>{filas}</table>")
            elif rol in FW:
                s.append(f'<p class="fw {FW[rol]}">{t}</p>')
            elif rol == "pie":
                s.append(f'<p class="nota">{t}</p>')
            else:
                s.append(f"<p>{t}</p>")
        s.append("</section>")
    s.append("</body></html>")
    return "\n".join(s) + "\n"


def _html_corrido(d: Documento) -> str:
    """El texto de lectura en HTML (sin mobiliario, con los párrafos partidos entre páginas unidos)."""
    out = []
    for rol, t in corrido(d, rico=False):
        if rol == "titulo":
            m = re.match(r"^(#+)\s*(.*)$", t)
            nivel = min(6, len(m.group(1))) if m else 2
            out.append(f"<h{nivel}>{escape(m.group(2) if m else t)}</h{nivel}>")
        elif rol == "lista":
            out.append(f"<ul><li>{escape(t[2:])}</li></ul>")
        elif rol in ("nota", "pie"):
            out.append(f'<p class="nota">{escape(t.strip("*"))}</p>')
        elif rol == "notas":
            out.append("<hr>")
        elif rol == "verso":
            out.append('<p class="verso">' + "<br>".join(escape(v) for v in t.split("  \n")) + "</p>")
        elif rol == "figura":
            m = re.match(r"!\[(.*?)\]\((.*?)\)", t)
            if m:
                out.append(f'<figure><img src="{escape(m.group(2))}" alt={quoteattr(m.group(1))} loading="lazy"></figure>')
        elif rol == "tabla":
            filas = [f.strip("|").split(" | ") for f in t.splitlines() if not f.startswith("|---")]
            out.append("<table>" + "".join("<tr>" + "".join(f"<td>{escape(c.strip())}</td>" for c in f) + "</tr>" for f in filas) + "</table>")
        else:
            out.append(f"<p>{escape(t)}</p>")
    return "\n".join(out)


def a_json(d: Documento) -> str:
    paginas = []
    for p in d.paginas:
        js = asdict(p)
        for l, lj in zip(p.lineas, js["lineas"]):
            lj["dudas"] = [[k, c] for k, c in enumerate(l.confs) if c < d.umbral_duda and not l.texto[k:k + 1].isspace()] if len(l.confs) == len(l.texto) else []
        js["bloques"] = [{k: v for k, v in b.items()} for b in bloques(p)]
        er = _estructura_rica(p)
        if er is not None:
            js["estructura_rica"] = er.a_json(p.como_cadena(), p.estructura)
        paginas.append(js)
    return json.dumps({"nombre": d.nombre, "umbral_duda": d.umbral_duda, "paginas": paginas}, ensure_ascii=False, indent=1)


def _c(b):
    return " ".join(str(int(round(v))) for v in b)


# ---------------------------------------------------------------- hOCR
def hocr(d: Documento) -> str:
    s = ['<?xml version="1.0" encoding="UTF-8"?>', '<html xmlns="http://www.w3.org/1999/xhtml"><head><meta charset="utf-8"/>',
         '<meta name="ocr-system" content="amanuensis"/>',
         '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_wconf ocr_photo ocr_table"/></head><body>']
    for i, p in enumerate(d.paginas):
        s.append(f'<div class="ocr_page" id="page_{i+1}" title="bbox 0 0 {p.ancho} {p.alto}; ppageno {i}">')
        hechas = set()
        for k, b in enumerate(bloques(p)):
            cajas = [p.lineas[j].caja for j in b["lineas"]]
            bb = [min(c[0] for c in cajas), min(c[1] for c in cajas), max(c[2] for c in cajas), max(c[3] for c in cajas)]
            clase = "ocr_table" if b["rol"] == "tabla" else "ocr_par"
            s.append(f'<div class="ocr_carea" id="block_{i+1}_{k+1}" title="bbox {_c(bb)}"><p class="{clase}" lang="und" data-rol="{b["rol"]}">')
            for j in b["lineas"]:
                hechas.add(j)
                s.append(_hocr_linea(i, j, p.lineas[j]))
            s.append("</p></div>")
        for j, l in enumerate(p.lineas):  # líneas fuera de los bloques (dentro de figuras, pies)
            if j not in hechas and l.texto.strip():
                s.append(_hocr_linea(i, j, l))
        for f in p.figuras:
            s.append(f'<div class="ocr_photo" id="{f["id"]}" title="bbox {_c(f["caja"])}; image {f["fichero"]}"></div>')
        s.append("</div>")
    s.append("</body></html>")
    return "\n".join(s) + "\n"


def _hocr_linea(i, j, l):
    o = [f'<span class="ocr_line" id="line_{i+1}_{j+1}" title="bbox {_c(l.caja)}; x_wconf {int(100*l.conf)}">']
    for k, (w, b, cs) in enumerate(l.palabras()):
        extra = f"; x_wconf {int(100 * min(cs))}; x_confs {' '.join(str(int(round(100 * c))) for c in cs)}" if cs else ""
        o.append(f'<span class="ocrx_word" id="word_{i+1}_{j+1}_{k+1}" title="bbox {_c(b)}{extra}">{escape(w)}</span>')
    o.append("</span>")
    return "".join(o)


# ---------------------------------------------------------------- ALTO v4
def alto(d: Documento) -> str:
    s = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
         'xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/alto/v4/alto-4-2.xsd">',
         f'<Description><MeasurementUnit>pixel</MeasurementUnit><sourceImageInformation><fileName>{escape(d.nombre)}</fileName>'
         '</sourceImageInformation><OCRProcessing ID="eg2"><ocrProcessingStep><processingSoftware><softwareName>amanuensis</softwareName>'
         '</processingSoftware></ocrProcessingStep></OCRProcessing></Description>']
    roles = sorted({b["rol"] for p in d.paginas for b in bloques(p)})
    if roles:
        s.append("<Tags>" + "".join(f'<LayoutTag ID="rol_{r}" LABEL="{r}"/>' for r in roles) + "</Tags>")
    s.append("<Layout>")
    for i, p in enumerate(d.paginas):
        s.append(f'<Page ID="p{i+1}" PHYSICAL_IMG_NR="{i+1}" WIDTH="{p.ancho}" HEIGHT="{p.alto}"><PrintSpace HPOS="0" VPOS="0" WIDTH="{p.ancho}" HEIGHT="{p.alto}">')
        bs = bloques(p) or ([{"rol": "texto", "lineas": list(range(len(p.lineas)))}] if p.lineas else [])
        for k, b in enumerate(bs):
            if b["rol"] == "tabla" and b.get("celdas"):
                cj = _caja_de(p, b["lineas"])
                s.append(f'<ComposedBlock ID="p{i+1}_b{k+1}" TYPE="table" {_hv(cj)} TAGREFS="rol_tabla">')
                for r, fila in enumerate(b["celdas"]):
                    for c, celda in enumerate(fila):
                        if celda:
                            s.append(f'<TextBlock ID="p{i+1}_b{k+1}_f{r+1}c{c+1}" {_hv(_caja_de(p, celda))}>')
                            s += [_alto_linea(i, j, p.lineas[j], d.umbral_duda) for j in celda]
                            s.append("</TextBlock>")
                s.append("</ComposedBlock>")
                continue
            if not b["lineas"]:
                continue
            s.append(f'<TextBlock ID="p{i+1}_b{k+1}" {_hv(_caja_de(p, b["lineas"]))} TAGREFS="rol_{b["rol"]}">')
            s += [_alto_linea(i, j, p.lineas[j], d.umbral_duda) for j in b["lineas"]]
            s.append("</TextBlock>")
        for f in p.figuras:
            s.append(f'<Illustration ID="{f["id"]}" {_hv(f["caja"])} TYPE="{f["tipo"]}" FILEID={quoteattr(f["fichero"])}/>')
        s.append("</PrintSpace></Page>")
    s.append("</Layout></alto>")
    return "\n".join(s) + "\n"


def _caja_de(p, idx):
    cs = [p.lineas[j].caja for j in idx]
    return [min(c[0] for c in cs), min(c[1] for c in cs), max(c[2] for c in cs), max(c[3] for c in cs)]


def _hv(c):
    x0, y0, x1, y1 = (int(round(v)) for v in c)
    return f'HPOS="{x0}" VPOS="{y0}" WIDTH="{x1-x0}" HEIGHT="{y1-y0}"'


def _estilo_palabras(l, n) -> list:
    """Bits de estilo de cada palabra (los que tiene más de la mitad de sus letras)."""
    if not l.estilos:
        return [0] * n
    bs, t, out, k = l.bits_por_letra(), l.texto, [], 0
    while k < len(t):
        if t[k] == " ":
            k += 1
            continue
        j = k
        while j < len(t) and t[j] != " ":
            j += 1
        w = bs[k:j]
        out.append(sum(m for m in ESTILO.values() if sum(1 for b in w if b & m) * 2 > len(w)))
        k = j
    return (out + [0] * n)[:n]


def _alto_linea(i, j, l, umbral):
    o = [f'<TextLine ID="p{i+1}_l{j+1}" {_hv(l.caja)}>']
    pal = l.palabras()
    est = _estilo_palabras(l, len(pal))
    for k, (w, b, cs) in enumerate(pal):
        wc = min(cs) if cs else l.conf
        cc = f' CC="{" ".join(str(min(9, max(0, int(round((1 - c) * 9))))) for c in cs)}"' if cs else ""
        st = " ".join(v for m, v in ESTILO_ALTO.items() if est[k] & m)
        st = f' STYLE="{st}"' if st else ""
        o.append(f'<String ID="p{i+1}_l{j+1}_w{k+1}" {_hv(b)} CONTENT={quoteattr(w)} WC="{wc:.2f}"{cc}{st}/>')
        if k + 1 < len(pal):
            o.append("<SP/>")
    o.append("</TextLine>")
    return "".join(o)


# ---------------------------------------------------------------- PAGE 2019
def _pts(b):
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    return f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"


TIPO_PAGE = {"texto": "paragraph", "lista": "list-label", "margen": "marginalia", "cabecera": "header", "numero": "page-number",
             "signatura": "signature-mark", "pie_pagina": "footer", "titulo": "heading", "nota": "footnote", "pie": "caption",
             "capitular": "drop-capital", "formula": "other", "musica": "other", "artefacto": "other", "figura": "other"}


def _page_linea(rid, j, l):
    y = int(round(l.caja[1] + 0.8 * (l.caja[3] - l.caja[1])))
    coords = " ".join(f"{int(round(a))},{int(round(b))}" for a, b in l.poligono) if l.poligono else _pts(l.caja)
    base = " ".join(f"{int(round(a))},{int(round(b))}" for a, b in l.base) if l.base else f"{int(l.caja[0])},{y} {int(l.caja[2])},{y}"
    tipo = f' custom="structure {{type:{l.tipo};}}"' if l.tipo else ""
    pal = ""
    if l.xs and len(l.xs) == len(l.texto):
        pal = "".join(f'<Word id="{rid}_l{j+1}_w{k+1}"><Coords points="{_pts(b)}"/><TextEquiv conf="{(min(cs) if cs else l.conf):.3f}">'
                      f'<Unicode>{escape(w)}</Unicode></TextEquiv></Word>' for k, (w, b, cs) in enumerate(l.palabras()))
    sty = ""
    if l.estilos:
        bs = [b for b, ch in zip(l.bits_por_letra(), l.texto) if not ch.isspace()]
        m = sum(v for v in ESTILO.values() if bs and sum(1 for b in bs if b & v) * 2 > len(bs))
        at = {1: 'bold="true"', 2: 'italic="true"', 4: 'smallCaps="true"', 8: 'underlined="true"'}
        sty = " ".join(a for k, a in at.items() if m & k)
        sty = f"<TextStyle {sty}/>" if sty else ""
    return (f'<TextLine id="{rid}_l{j+1}"{tipo}><Coords points="{coords}"/><Baseline points="{base}"/>{pal}'
            f'<TextEquiv conf="{l.conf:.3f}"><Unicode>{escape(l.texto)}</Unicode></TextEquiv>{sty}</TextLine>')


def page(d: Documento, solo: int | None = None) -> str:
    """PAGE XML 2019: una región por bloque (TextRegion con su tipo, TableRegion con celdas por TableCellRole), orden de
    lectura de las regiones, líneas con polígono, línea base y palabras con confianza. Con varias páginas se
    concatenan documentos PcGts separados por un comentario (la orden escribe un fichero por página)."""
    docs = []
    for i, p in enumerate(d.paginas):
        if solo is not None and i != solo:
            continue
        s = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">',
             '<Metadata><Creator>amanuensis</Creator><Created>1970-01-01T00:00:00</Created><LastChange>1970-01-01T00:00:00</LastChange></Metadata>',
             f'<Page imageFilename={quoteattr(d.nombre)} imageWidth="{p.ancho}" imageHeight="{p.alto}">']
        bs = [b for b in bloques(p) if b["lineas"]]
        usadas = {j for b in bs for j in b["lineas"]}
        sueltas = [j for j, l in enumerate(p.lineas) if j not in usadas and l.texto.strip()]
        if sueltas:
            bs.append({"rol": "texto", "lineas": sueltas, "suelto": True})
        regiones = []
        for k, b in enumerate(bs):
            rid = f"r{k+1}"
            regiones.append(rid)
            if b["rol"] == "tabla" and b.get("celdas"):
                n = max(len(f) for f in b["celdas"])
                s.append(f'<TableRegion id="{rid}" rows="{len(b["celdas"])}" columns="{n}"><Coords points="{_pts(_caja_de(p, b["lineas"]))}"/>')
                for r, fila in enumerate(b["celdas"]):
                    for c, celda in enumerate(fila):
                        if not celda:
                            continue
                        cid = f"{rid}_f{r+1}c{c+1}"
                        s.append(f'<TextRegion id="{cid}" type="paragraph"><Roles><TableCellRole rowIndex="{r}" columnIndex="{c}"/></Roles>'
                                 f'<Coords points="{_pts(_caja_de(p, celda))}"/>' + "".join(_page_linea(cid, j, p.lineas[j]) for j in celda) +
                                 f'<TextEquiv><Unicode>{escape(unir([p.lineas[j] for j in celda]))}</Unicode></TextEquiv></TextRegion>')
                s.append("</TableRegion>")
                continue
            s.append(f'<TextRegion id="{rid}" type="{TIPO_PAGE.get(b["rol"], "paragraph")}"><Coords points="{_pts(_caja_de(p, b["lineas"]))}"/>')
            s += [_page_linea(rid, j, p.lineas[j]) for j in b["lineas"]]
            s.append(f'<TextEquiv><Unicode>{escape(chr(10).join(p.lineas[j].texto for j in b["lineas"]))}</Unicode></TextEquiv></TextRegion>')
        for f in p.figuras:
            pts = _pts(f["caja"])
            com = escape(f'image {f["fichero"]}; alt {f["alt"]}' + (f'; caption {f["pie"]}' if f["pie"] else ""))
            regiones.append(f["id"])
            if f["tipo"] in ("figura", "ilustracion"):
                s.append(f'<ImageRegion id="{f["id"]}" comments="{com}"><Coords points="{pts}"/></ImageRegion>')
            elif f["tipo"] == "formula":
                s.append(f'<MathsRegion id="{f["id"]}" comments="{com}"><Coords points="{pts}"/></MathsRegion>')
            else:
                t = {"capitular": "decoration", "sello": "stamp", "sello_tinta": "stamp"}.get(f["tipo"], "other")
                s.append(f'<GraphicRegion id="{f["id"]}" type="{t}" comments="{com}"><Coords points="{pts}"/></GraphicRegion>')
        if regiones:
            ro = "".join(f'<RegionRefIndexed index="{k}" regionRef="{r}"/>' for k, r in enumerate(regiones))
            # el orden de lectura va justo después de <Page> según el esquema
            s.insert(4, f'<ReadingOrder><OrderedGroup id="ro1">{ro}</OrderedGroup></ReadingOrder>')
        s.append("</Page></PcGts>")
        docs.append("\n".join(s))
    return "\n<!-- siguiente página -->\n".join(docs) + "\n"


# ---------------------------------------------------------------- TEI
def _tei_linea(l, umbral) -> str:
    o = []
    if l.estilos:
        for t, cert, b in tramos_ricos(l.letras(umbral)):
            x = f'<unclear reason="illegible" cert="{cert:.2f}">{escape(t)}</unclear>' if cert is not None else escape(t)
            o.append(f'<hi rend="{_bits_rend(b)}">{x}</hi>' if b else x)
        return "".join(o)
    for t, cert in l.tramos(umbral):
        o.append(f'<unclear reason="illegible" cert="{cert:.2f}">{escape(t)}</unclear>' if cert is not None else escape(t))
    return "".join(o)


def tei(d: Documento) -> str:
    """TEI con <facsimile> (zona de cada línea, de cada región y de cada figura) y el texto por página: párrafos <p>,
    títulos <head>, listas, notas al margen <note place="margin"> en su ancla, notas al pie, cabeceras y números de
    página <fw>, tablas <table>/<row>/<cell>, figuras con pie; cada línea con <lb facs> y las letras dudosas en
    <unclear cert>."""
    s = ['<?xml version="1.0" encoding="UTF-8"?>', '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>',
         f'<titleStmt><title>{escape(d.nombre)}</title></titleStmt><publicationStmt><p>Transcripción automática con Amanuensis</p></publicationStmt>',
         f'<sourceDesc><p>{escape(d.nombre)}</p></sourceDesc></fileDesc>'
         f'<encodingDesc><p>Letras con probabilidad menor que {d.umbral_duda:g} en &lt;unclear cert&gt; (probabilidad mínima del tramo).</p></encodingDesc>'
         '</teiHeader><facsimile>']
    for i, p in enumerate(d.paginas):
        s.append(f'<surface xml:id="f{i+1}" ulx="0" uly="0" lrx="{p.ancho}" lry="{p.alto}">')
        for j, l in enumerate(p.lineas):
            x0, y0, x1, y1 = (int(round(v)) for v in l.caja)
            pts = f' points="{" ".join(f"{int(round(a))},{int(round(b))}" for a, b in l.poligono)}"' if l.poligono else ""
            s.append(f'<zone xml:id="f{i+1}_l{j+1}" ulx="{x0}" uly="{y0}" lrx="{x1}" lry="{y1}"{pts}/>')
        for f in p.figuras:
            x0, y0, x1, y1 = (int(round(v)) for v in f["caja"])
            s.append(f'<zone xml:id="{f["id"]}" type="{f["tipo"]}" ulx="{x0}" uly="{y0}" lrx="{x1}" lry="{y1}"><graphic url={quoteattr(f["fichero"])}/></zone>')
        s.append("</surface>")
    s.append("</facsimile><text><body>")
    tipo_tei = {"figura": "figure", "ilustracion": "illustration", "capitular": "initial", "sello": "seal",
                "sello_tinta": "stamp", "formula": "formula"}
    for i, p in enumerate(d.paginas):
        s.append(f'<pb n="{i+1}" facs="#f{i+1}"/>')
        er = _estructura_rica(p)
        if er is not None:
            s.append(er.a_tei(p.como_cadena(), p.estructura, id_pag=f"f{i+1}"))
            continue
        lb = lambda j: f'<lb facs="#f{i+1}_l{j+1}"/>{_tei_linea(p.lineas[j], d.umbral_duda)}'
        s.append("<div>")
        seq = secuencia(p)
        for tipo, x in seq:
            if tipo == "f":
                head = f"<head>{escape(x['pie'])}</head>" if x["pie"] else ""
                ic = "".join(f'<term ref="http://iconclass.org/{escape(y["notacion"])}">{escape(y["en"])}</term>' for y in x.get("iconclass", [])[:3])
                ic = f'<note type="iconclass" subtype="sugerencia">{ic}</note>' if ic else ""
                s.append(f'<figure type="{tipo_tei.get(x["tipo"], "figure")}" facs="#{x["id"]}" xml:id="{x["id"]}_fig">'
                         f'<graphic url={quoteattr(x["fichero"])}/>{head}<figDesc>{escape(x["alt"])}</figDesc>{ic}</figure>')
                continue
            rol, ls = x["rol"], x["lineas"]
            if rol == "margen":
                ref = f' target="#f{i+1}_l{x["ancla"]+1}"' if "ancla" in x else ""
                s.append(f'<note place="margin"{ref}>' + "".join(lb(j) for j in ls) + "</note>")
            elif rol in FW:
                s.append(f'<fw type="{FW[rol]}">' + "".join(lb(j) for j in ls) + "</fw>")
            elif rol == "titulo":
                s.append("<head>" + "".join(lb(j) for j in ls) + "</head>")
            elif rol == "lista":
                s.append("<list><item>" + "".join(lb(j) for j in ls) + "</item></list>")
            elif rol == "nota":
                s.append('<note place="foot">' + "".join(lb(j) for j in ls) + "</note>")
            elif rol == "tabla" and x.get("celdas"):
                s.append("<table>" + "".join("<row>" + "".join("<cell>" + "".join(lb(j) for j in c) + "</cell>" for c in f) + "</row>"
                                             for f in x["celdas"]) + "</table>")
            elif rol == "formula":
                s.append('<figure type="formula"><p>' + "".join(lb(j) for j in ls) + "</p></figure>")
            elif rol in ("musica", "artefacto", "figura", "pie"):
                s.append(f'<ab type="{rol}">' + "".join(lb(j) for j in ls) + "</ab>")
            else:
                s.append("<p>" + "".join(lb(j) for j in ls) + "</p>")
        s.append("</div>")
    s.append("</body></text></TEI>")
    return "\n".join(s) + "\n"


def spdf(d: Documento) -> str:
    """SPDF (~/Developer/spdf): se usa su biblioteca si está instalada; el contrato es el JSON de Amanuensis."""
    try:
        import spdf as _spdf  # noqa: F401
    except ImportError:
        raise SystemExit("la salida SPDF necesita el paquete «spdf» (aún no publicado); usa --formato json")
    return _spdf.desde_amanuensis(json.loads(a_json(d)))
