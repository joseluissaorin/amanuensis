"""Documento leído y sus salidas: txt, md, json, hocr, alto (v4), page (PAGE 2019), tei."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
import re
from xml.sax.saxutils import escape as _escape, quoteattr as _quoteattr

_NO_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff\ud800-\udfff]")


def escape(t) -> str:  # quita los caracteres que XML 1.0 no admite (algunas capas de texto de PDF los traen)
    return _escape(_NO_XML.sub("", str(t)))


def quoteattr(t) -> str:
    return _quoteattr(_NO_XML.sub("", str(t)))

FORMATOS = ("md", "txt", "json", "html", "hocr", "alto", "page", "tei", "spdf")
EXTENSION = {"md": ".md", "txt": ".txt", "json": ".json", "html": ".html", "hocr": ".hocr", "alto": ".alto.xml",
             "page": ".page.xml", "tei": ".tei.xml", "spdf": ".spdf"}


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

    def palabras(self):
        """[(texto, caja)] a partir de las posiciones de los caracteres (o repartiendo el ancho)."""
        x0, y0, x1, y1 = self.caja
        t = self.texto
        xs = self.xs if len(self.xs) == len(t) and t else None
        if xs is None:
            n = max(1, len(t))
            xs = [x0 + (i + 0.5) * (x1 - x0) / n for i in range(len(t))]
        paso = (x1 - x0) / max(1, len(t))
        out, i = [], 0
        while i < len(t):
            if t[i] == " ":
                i += 1
                continue
            j = i
            while j < len(t) and t[j] != " ":
                j += 1
            out.append((t[i:j], [max(x0, xs[i] - paso / 2), y0, min(x1, xs[j - 1] + paso / 2), y1]))
            i = j
        return out


@dataclass
class Pagina:
    ancho: int
    alto: int
    lineas: list
    fuente: str = "ocr"  # ocr | texto (capa de texto del PDF)
    modelo: str = ""
    segundos: float = 0.0
    zonas: list = field(default_factory=list)  # [{"clase", "caja", "conf"}] (maquetación)
    figuras: list = field(default_factory=list)  # imágenes extraídas (figuras.py)

    @property
    def texto(self):
        return "\n".join(l.texto for l in self.lineas)


@dataclass
class Documento:
    nombre: str
    paginas: list
    imagenes: dict = field(default_factory=dict, repr=False)  # ruta relativa -> imagen PIL (recortes a guardar)

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


def _corrido(l) -> bool:
    return bool(l.texto.strip()) and not l.en_figura and not l.pie_de


def txt(d: Documento) -> str:
    return "\n\f\n".join("\n".join(l.texto for l in p.lineas if not l.en_figura) for p in d.paginas) + "\n"


def parrafos(p: Pagina, con_indice: bool = False) -> list:
    """Une líneas en párrafos por hueco vertical, sangría y fin de frase; quita guiones de corte.
    Deja fuera el texto de dentro de las figuras y sus pies (van con la figura). Con con_indice, devuelve
    (texto, índice de su primera línea)."""
    idx = [i for i, l in enumerate(p.lineas) if _corrido(l)]
    ls = [p.lineas[i] for i in idx]
    if not ls:
        return []
    hs = sorted(l.caja[3] - l.caja[1] for l in ls)
    hmed = hs[len(hs) // 2]
    izq = sorted(l.caja[0] for l in ls)[len(ls) // 4]
    out, act, ini = [], ls[0].texto.strip(), idx[0]
    for k, (a, b) in enumerate(zip(ls, ls[1:])):
        hueco = b.caja[1] - a.caja[3]
        ha, hb = a.caja[3] - a.caja[1], b.caja[3] - b.caja[1]
        nuevo = abs(ha - hb) > 0.25 * max(ha, hb) or hueco > 0.8 * hmed or b.caja[1] < a.caja[1] - hmed or (b.caja[0] - izq > 1.5 * hmed and a.texto.rstrip()[-1:] in ".:!?»\"")
        if nuevo:
            out.append((act, ini)); act = b.texto.strip(); ini = idx[k + 1]
        elif act[-1:] in "-¬⸗" and len(act) > 1 and act[-2].isalpha():
            act = act[:-1] + b.texto.strip()
        else:
            act += " " + b.texto.strip()
    out.append((act, ini))
    return out if con_indice else [t for t, _ in out]


def ancla(p: Pagina, fig) -> int:
    """Índice de la línea (en orden de lectura) delante de la cual va la figura: la primera línea de texto
    corrido que empieza por debajo del borde superior de la figura y la solapa en horizontal; si no hay, al final."""
    c = fig["caja"]
    for i, l in enumerate(p.lineas):
        if _corrido(l) and l.caja[1] >= c[1] and min(l.caja[2], c[2]) - max(l.caja[0], c[0]) > 0:
            return i
    return len(p.lineas)


def secuencia(p: Pagina, capitulares: bool = False):
    """Párrafos y figuras intercalados en orden: [("p", texto) | ("f", figura)]."""
    figs = [f for f in p.figuras if capitulares or f["tipo"] != "capitular"]
    anclas = sorted(((ancla(p, f), n, f) for n, f in enumerate(figs)), key=lambda t: (t[0], t[1]))
    out, k = [], 0
    for texto, ini in parrafos(p, True):
        while k < len(anclas) and anclas[k][0] <= ini:
            out.append(("f", anclas[k][2])); k += 1
        out.append(("p", texto))
    out += [("f", a[2]) for a in anclas[k:]]
    return out


def md(d: Documento) -> str:
    partes = []
    for i, p in enumerate(d.paginas):
        if len(d.paginas) > 1:
            partes.append(f"<!-- página {i + 1} -->")
        for tipo, x in secuencia(p):
            if tipo == "p":
                partes.append(x)
            else:
                alt = x["alt"].replace("[", "(").replace("]", ")")
                partes.append(f"![{alt}]({x['fichero']})" + (f"\n\n*{x['pie']}*" if x["pie"] else ""))
    return "\n\n".join(partes) + "\n"


def html(d: Documento) -> str:
    s = ['<!doctype html>', '<html lang="und"><head><meta charset="utf-8">', f"<title>{escape(d.nombre)}</title>",
         '<meta name="generator" content="amanuensis">',
         "<style>body{max-width:44rem;margin:2rem auto;font:1.05rem/1.55 Georgia,serif;padding:0 1rem}figure{margin:1.5rem 0}"
         "figure img{max-width:100%}figcaption{font-size:.9rem;color:#555}</style></head><body>"]
    for i, p in enumerate(d.paginas):
        s.append(f'<section class="pagina" id="p{i + 1}">')
        for tipo, x in secuencia(p):
            if tipo == "p":
                s.append(f"<p>{escape(x)}</p>")
            else:
                cap = f"<figcaption>{escape(x['pie'])}</figcaption>" if x["pie"] else ""
                s.append(f'<figure id="{x["id"]}" class="{x["tipo"]}"><img src="{escape(x["fichero"])}" alt={quoteattr(x["alt"])} loading="lazy">{cap}</figure>')
        s.append("</section>")
    s.append("</body></html>")
    return "\n".join(s) + "\n"


def a_json(d: Documento) -> str:
    return json.dumps({"nombre": d.nombre, "paginas": [asdict(p) for p in d.paginas]}, ensure_ascii=False, indent=1)


def _c(b):
    return " ".join(str(int(round(v))) for v in b)


def hocr(d: Documento) -> str:
    s = ['<?xml version="1.0" encoding="UTF-8"?>', '<html xmlns="http://www.w3.org/1999/xhtml"><head><meta charset="utf-8"/>',
         '<meta name="ocr-system" content="amanuensis"/><meta name="ocr-capabilities" content="ocr_page ocr_line ocrx_word"/></head><body>']
    for i, p in enumerate(d.paginas):
        s.append(f'<div class="ocr_page" id="page_{i+1}" title="bbox 0 0 {p.ancho} {p.alto}; ppageno {i}">')
        for j, l in enumerate(p.lineas):
            s.append(f'<span class="ocr_line" id="line_{i+1}_{j+1}" title="bbox {_c(l.caja)}; x_wconf {int(100*l.conf)}">')
            for k, (w, b) in enumerate(l.palabras()):
                s.append(f'<span class="ocrx_word" id="word_{i+1}_{j+1}_{k+1}" title="bbox {_c(b)}">{escape(w)}</span>')
            s.append("</span>")
        for f in p.figuras:
            s.append(f'<div class="ocr_photo" id="{f["id"]}" title="bbox {_c(f["caja"])}; image {f["fichero"]}"></div>')
        s.append("</div>")
    s.append("</body></html>")
    return "\n".join(s) + "\n"


def alto(d: Documento) -> str:
    s = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
         'xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/alto/v4/alto-4-2.xsd">',
         f'<Description><MeasurementUnit>pixel</MeasurementUnit><sourceImageInformation><fileName>{escape(d.nombre)}</fileName>'
         '</sourceImageInformation><OCRProcessing ID="eg2"><ocrProcessingStep><processingSoftware><softwareName>amanuensis</softwareName>'
         '</processingSoftware></ocrProcessingStep></OCRProcessing></Description><Layout>']
    for i, p in enumerate(d.paginas):
        s.append(f'<Page ID="p{i+1}" PHYSICAL_IMG_NR="{i+1}" WIDTH="{p.ancho}" HEIGHT="{p.alto}"><PrintSpace HPOS="0" VPOS="0" WIDTH="{p.ancho}" HEIGHT="{p.alto}">')
        s.append(f'<TextBlock ID="p{i+1}_b1">')
        for j, l in enumerate(p.lineas):
            x0, y0, x1, y1 = (int(round(v)) for v in l.caja)
            s.append(f'<TextLine ID="p{i+1}_l{j+1}" HPOS="{x0}" VPOS="{y0}" WIDTH="{x1-x0}" HEIGHT="{y1-y0}">')
            pal = l.palabras()
            for k, (w, b) in enumerate(pal):
                a0, b0, a1, b1 = (int(round(v)) for v in b)
                s.append(f'<String ID="p{i+1}_l{j+1}_w{k+1}" HPOS="{a0}" VPOS="{b0}" WIDTH="{a1-a0}" HEIGHT="{b1-b0}" CONTENT={quoteattr(w)} WC="{l.conf:.2f}"/>')
                if k + 1 < len(pal):
                    s.append("<SP/>")
            s.append("</TextLine>")
        s.append("</TextBlock>")
        for f in p.figuras:
            a0, b0, a1, b1 = (int(round(v)) for v in f["caja"])
            s.append(f'<Illustration ID="{f["id"]}" HPOS="{a0}" VPOS="{b0}" WIDTH="{a1-a0}" HEIGHT="{b1-b0}" TYPE="{f["tipo"]}" FILEID={quoteattr(f["fichero"])}/>')
        s.append("</PrintSpace></Page>")
    s.append("</Layout></alto>")
    return "\n".join(s) + "\n"


def _pts(b):
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    return f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"


def page(d: Documento, solo: int | None = None) -> str:
    """PAGE XML 2019; con varias páginas, una <Page> por página dentro del mismo documento no es válido,
    así que se concatenan documentos PcGts separados por un comentario (usar --por-pagina si hace falta)."""
    docs = []
    for i, p in enumerate(d.paginas):
        if solo is not None and i != solo:
            continue
        s = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">',
             '<Metadata><Creator>amanuensis</Creator><Created>1970-01-01T00:00:00</Created><LastChange>1970-01-01T00:00:00</LastChange></Metadata>',
             f'<Page imageFilename={quoteattr(d.nombre)} imageWidth="{p.ancho}" imageHeight="{p.alto}">']
        if p.lineas:
            xs0 = min(l.caja[0] for l in p.lineas); ys0 = min(l.caja[1] for l in p.lineas)
            xs1 = max(l.caja[2] for l in p.lineas); ys1 = max(l.caja[3] for l in p.lineas)
            s.append(f'<ReadingOrder><OrderedGroup id="ro1"><RegionRefIndexed index="0" regionRef="r1"/></OrderedGroup></ReadingOrder>')
            s.append(f'<TextRegion id="r1"><Coords points="{_pts([xs0, ys0, xs1, ys1])}"/>')
            for j, l in enumerate(p.lineas):
                y = int(round(l.caja[1] + 0.8 * (l.caja[3] - l.caja[1])))
                coords = " ".join(f"{int(round(a))},{int(round(b))}" for a, b in l.poligono) if l.poligono else _pts(l.caja)
                base = " ".join(f"{int(round(a))},{int(round(b))}" for a, b in l.base) if l.base else f"{int(l.caja[0])},{y} {int(l.caja[2])},{y}"
                tipo = f' custom="structure {{type:{l.tipo};}}"' if l.tipo else ""
                s.append(f'<TextLine id="r1_l{j+1}"{tipo}><Coords points="{coords}"/><Baseline points="{base}"/>'
                         f'<TextEquiv conf="{l.conf:.3f}"><Unicode>{escape(l.texto)}</Unicode></TextEquiv></TextLine>')
            s.append(f'<TextEquiv><Unicode>{escape(p.texto)}</Unicode></TextEquiv></TextRegion>')
        for f in p.figuras:
            pts = _pts(f["caja"])
            com = escape(f'image {f["fichero"]}; alt {f["alt"]}' + (f'; caption {f["pie"]}' if f["pie"] else ""))
            if f["tipo"] in ("figura", "ilustracion"):
                s.append(f'<ImageRegion id="{f["id"]}" comments="{com}"><Coords points="{pts}"/></ImageRegion>')
            elif f["tipo"] == "formula":
                s.append(f'<MathsRegion id="{f["id"]}" comments="{com}"><Coords points="{pts}"/></MathsRegion>')
            else:
                t = {"capitular": "decoration", "sello": "stamp", "sello_tinta": "stamp"}.get(f["tipo"], "other")
                s.append(f'<GraphicRegion id="{f["id"]}" type="{t}" comments="{com}"><Coords points="{pts}"/></GraphicRegion>')
        s.append("</Page></PcGts>")
        docs.append("\n".join(s))
    return "\n<!-- siguiente página -->\n".join(docs) + "\n"


def tei(d: Documento) -> str:
    """TEI con <facsimile> (zona de cada línea y de cada figura) y el texto por página: líneas con <lb facs>
    y figuras como <figure type><graphic url/><head>pie</head><figDesc>alt</figDesc></figure> en su sitio."""
    s = ['<?xml version="1.0" encoding="UTF-8"?>', '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>',
         f'<titleStmt><title>{escape(d.nombre)}</title></titleStmt><publicationStmt><p>Transcripción automática con Amanuensis</p></publicationStmt>',
         f'<sourceDesc><p>{escape(d.nombre)}</p></sourceDesc></fileDesc></teiHeader><facsimile>']
    for i, p in enumerate(d.paginas):
        s.append(f'<surface xml:id="f{i+1}" ulx="0" uly="0" lrx="{p.ancho}" lry="{p.alto}">')
        for j, l in enumerate(p.lineas):
            x0, y0, x1, y1 = (int(round(v)) for v in l.caja)
            s.append(f'<zone xml:id="f{i+1}_l{j+1}" ulx="{x0}" uly="{y0}" lrx="{x1}" lry="{y1}"/>')
        for f in p.figuras:
            x0, y0, x1, y1 = (int(round(v)) for v in f["caja"])
            s.append(f'<zone xml:id="{f["id"]}" type="{f["tipo"]}" ulx="{x0}" uly="{y0}" lrx="{x1}" lry="{y1}"><graphic url={quoteattr(f["fichero"])}/></zone>')
        s.append("</surface>")
    s.append("</facsimile><text><body><div>")
    tipo_tei = {"figura": "figure", "ilustracion": "illustration", "capitular": "initial", "sello": "seal",
                "sello_tinta": "stamp", "formula": "formula"}
    for i, p in enumerate(d.paginas):
        s.append(f'<pb n="{i+1}" facs="#f{i+1}"/>')
        porlinea = {}
        for f in p.figuras:
            porlinea.setdefault(ancla(p, f) if f["tipo"] != "capitular" else (f["palabra"] or {}).get("linea", ancla(p, f)), []).append(f)
        abierto = False
        for j, l in enumerate(p.lineas + [None]):
            for f in porlinea.get(j, []):
                if abierto:
                    s.append("</ab>"); abierto = False
                head = f"<head>{escape(f['pie'])}</head>" if f["pie"] else ""
                ic = "".join(f'<term ref="http://iconclass.org/{escape(x["notacion"])}">{escape(x["en"])}</term>' for x in f["iconclass"][:3])
                ic = f'<note type="iconclass" subtype="sugerencia">{ic}</note>' if ic else ""
                s.append(f'<figure type="{tipo_tei.get(f["tipo"], "figure")}" facs="#{f["id"]}" xml:id="{f["id"]}_fig">'
                         f'<graphic url={quoteattr(f["fichero"])}/>{head}<figDesc>{escape(f["alt"])}</figDesc>{ic}</figure>')
            if l is None or l.pie_de or l.en_figura:
                continue
            if not abierto:
                s.append("<ab>"); abierto = True
            s.append(f'<lb facs="#f{i+1}_l{j+1}"/>{escape(l.texto)}')
        if abierto:
            s.append("</ab>")
    s.append("</div></body></text></TEI>")
    return "\n".join(s) + "\n"


def spdf(d: Documento) -> str:
    """Enchufe para SPDF (~/Developer/spdf): se usa su biblioteca si está instalada."""
    try:
        import spdf as _spdf  # noqa: F401
    except ImportError:
        raise SystemExit("la salida SPDF necesita el paquete «spdf» (aún no publicado); usa --formato json")
    return _spdf.desde_amanuensis(json.loads(a_json(d)))  # contrato por acordar con el proyecto SPDF
