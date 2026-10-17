"""Documento leído y sus salidas: txt, md, json, hocr, alto (v4), page (PAGE 2019), tei."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from xml.sax.saxutils import escape, quoteattr

FORMATOS = ("md", "txt", "json", "hocr", "alto", "page", "tei", "spdf")
EXTENSION = {"md": ".md", "txt": ".txt", "json": ".json", "hocr": ".hocr", "alto": ".alto.xml",
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

    @property
    def texto(self):
        return "\n".join(l.texto for l in self.lineas)


@dataclass
class Documento:
    nombre: str
    paginas: list

    def exportar(self, formato: str) -> str:
        return {"txt": txt, "md": md, "json": a_json, "hocr": hocr, "alto": alto, "page": page, "tei": tei,
                "spdf": spdf}[formato](self)


def txt(d: Documento) -> str:
    return "\n\f\n".join(p.texto for p in d.paginas) + "\n"


def parrafos(p: Pagina) -> list[str]:
    """Une líneas en párrafos por hueco vertical, sangría y fin de frase; quita guiones de corte."""
    ls = [l for l in p.lineas if l.texto.strip()]
    if not ls:
        return []
    hs = sorted(l.caja[3] - l.caja[1] for l in ls)
    hmed = hs[len(hs) // 2]
    izq = sorted(l.caja[0] for l in ls)[len(ls) // 4]
    out, act = [], ls[0].texto.strip()
    for a, b in zip(ls, ls[1:]):
        hueco = b.caja[1] - a.caja[3]
        ha, hb = a.caja[3] - a.caja[1], b.caja[3] - b.caja[1]
        nuevo = abs(ha - hb) > 0.25 * max(ha, hb) or hueco > 0.8 * hmed or b.caja[1] < a.caja[1] - hmed or (b.caja[0] - izq > 1.5 * hmed and a.texto.rstrip()[-1:] in ".:!?»\"")
        if nuevo:
            out.append(act); act = b.texto.strip()
        elif act[-1:] in "-¬⸗" and len(act) > 1 and act[-2].isalpha():
            act = act[:-1] + b.texto.strip()
        else:
            act += " " + b.texto.strip()
    out.append(act)
    return out


def md(d: Documento) -> str:
    partes = []
    for i, p in enumerate(d.paginas):
        if len(d.paginas) > 1:
            partes.append(f"<!-- página {i + 1} -->")
        partes += parrafos(p)
    return "\n\n".join(partes) + "\n"


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
        s.append("</TextBlock></PrintSpace></Page>")
    s.append("</Layout></alto>")
    return "\n".join(s) + "\n"


def _pts(b):
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    return f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"


def page(d: Documento) -> str:
    """PAGE XML 2019; con varias páginas, una <Page> por página dentro del mismo documento no es válido,
    así que se concatenan documentos PcGts separados por un comentario (usar --por-pagina si hace falta)."""
    docs = []
    for i, p in enumerate(d.paginas):
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
        s.append("</Page></PcGts>")
        docs.append("\n".join(s))
    return "\n<!-- siguiente página -->\n".join(docs) + "\n"


def tei(d: Documento) -> str:
    s = ['<?xml version="1.0" encoding="UTF-8"?>', '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>',
         f'<titleStmt><title>{escape(d.nombre)}</title></titleStmt><publicationStmt><p>Transcripción automática con amanuensis</p></publicationStmt>',
         f'<sourceDesc><p>{escape(d.nombre)}</p></sourceDesc></fileDesc></teiHeader><facsimile>']
    for i, p in enumerate(d.paginas):
        s.append(f'<surface xml:id="f{i+1}" ulx="0" uly="0" lrx="{p.ancho}" lry="{p.alto}">')
        for j, l in enumerate(p.lineas):
            x0, y0, x1, y1 = (int(round(v)) for v in l.caja)
            s.append(f'<zone xml:id="f{i+1}_l{j+1}" ulx="{x0}" uly="{y0}" lrx="{x1}" lry="{y1}"/>')
        s.append("</surface>")
    s.append("</facsimile><text><body><div>")
    for i, p in enumerate(d.paginas):
        s.append(f'<pb n="{i+1}" facs="#f{i+1}"/>')
        s.append("<ab>")
        for j, l in enumerate(p.lineas):
            s.append(f'<lb facs="#f{i+1}_l{j+1}"/>{escape(l.texto)}')
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
