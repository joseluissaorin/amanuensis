"""Estructura rica sobre la salida de la cadena de página (agente «estructura», 18-10-2026).

1. Diccionarios: `entradas_diccionario(pagina)` parte las líneas (en orden de lectura, con caja y zona) en ENTRADAS por la
   geometría (sangría francesa o de primera línea, medida por columna) y el texto (lema seguido de coma o punto, número
   de acepción), y separa lema, variante, categoría gramatical, etimología, acepciones numeradas, ejemplos y remisiones
   (sin tipografía: el lector CTC no da negrita ni cursiva, así que los campos salen de patrones de texto).
2. Salidas: `a_tei(pagina)` (TEI P5 con <entry><form><gramGrp><sense><cit> para diccionarios, <figure> enlazada con el
   texto vecino en manuscritos iluminados, <div type="panel"> con <said>/<note type="caption"> en cómics), `a_markdown`
   y `a_json` equivalentes.
3. detectar_tipo(pagina): «diccionario», «comic» o «texto» (heurística; la sonda de tipo de página aprendida irá en
   pagina["tipo"] cuando la maquetación tenga las cabezas de «estructura»).
4. Estilo por palabra (cabeza tipográfica; pagina["estilos"] alineado con las líneas): linea_tei (<hi rend>),
   linea_html, linea_markdown y alto_estilos (TextStyle FONTSTYLE/FONTCOLOR).
Copia autocontenida de maqueta/estructura.py del repo eg2-ocr (allí están la medida y la línea de órdenes).
"""
from __future__ import annotations

import ast
import json
import re
import unicodedata
from xml.sax.saxutils import escape


GRAM = (r"(?:s\.\s?m\.|s\.\s?f\.|n\.\s?m\.|n\.\s?f\.|adj\.|adv\.|v\.\s?a\.|v\.\s?n\.|v\.\s?r\.|v\.\s?tr\.|v\.\s?intr\.|v\.\s?pron\.|"
        r"v\.\s?t\.|v\.\s?i\.|tr\.|intr\.|prnl\.|loc\.\s?\w+\.|interj\.|prep\.|conj\.|pron\.|part\.|m\.|f\.|n\.|v\.|a\.|s\.|agg\.|avv\.|Subst\.)")
RE_GRAM = re.compile(r"^\s*(?:y\s|&\s)?(" + GRAM + r"(?:\s*(?:y|&|et|and)\s*" + GRAM + r")*)")
RE_NUM = re.compile(r"(?:(?<=\s)|^)(\d{1,2})[\.\)]\s|\|\|")
RE_ETIM = re.compile(r"^\s*[\(\[](?:Del?|Du|From|Aus|Dal|Do|De l')[^\)\]]{0,80}[\)\]]")
RE_REMITE = re.compile(r"^\s*(?:V\.|Véase|Voyez|See|Vgl\.|S\.|Lo mismo que|Same as)\s")
_PART = r"(?:de|del|la|las|los|le|les|l['’]|d['’]|à|a|y|e|et|of|the|und|von|di|da|do|des|du)"
RE_LEMA = re.compile(r"^\s*(?:☞\s*)?((?:[A-ZÀ-ÞÁÉÍÓÚÑ][^\s,\.;:\(\)]{0,40}|[a-zà-ÿ][^\s,\.;:\(\)]{1,40})"
                     r"(?:[\s\-'’](?:" + _PART + r"|[A-ZÀ-ÞÁÉÍÓÚÑ][^\s,\.;:\(\)]{0,30})){0,4})\s*([,\.:]|\s\()")


def clase(z):
    """Nombre de la clase de una zona (la cadena da «clase» como índice y el nombre en «tipo»)."""
    return z.get("tipo") if isinstance(z.get("clase"), int) else z.get("clase")


def _lista(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return []
    return v or []


def norm(t):
    t = unicodedata.normalize("NFD", (t or "").lower())
    return "".join(c for c in t if c.isalnum() and not unicodedata.combining(c))


def _cer(a, b):
    a, b = norm(a), norm(b)
    if not b:
        return 0.0 if not a else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(b)


# ---------------------------------------------------------------- diccionarios
def _parece_inicio(texto, geo):
    """¿Abre entrada esta línea? Lema en mayúsculas (fuerte) o, si la geometría lo dice, palabra inicial con coma o punto."""
    m = RE_LEMA.match(texto or "")
    if not m:
        return False
    lema = m.group(1)
    letras = [c for c in lema if c.isalpha()]
    if len(letras) >= 3 and all(c.isupper() for c in letras):
        return True
    if not geo:
        return False
    return lema[:1].isupper() or (len(lema.split()) == 1 and len(lema) <= 25) or texto.lstrip().startswith("☞")


def entradas_diccionario(pagina, zonas_texto=("MainZone", "Text", "List-item"), estilos=None, umbral=0.5):
    """pagina: salida de CadenaPagina ({"lineas": [{caja, texto, zona, zona_tipo}], "zonas": [...]}).
    estilos (opcional; la cabeza tipográfica, maqueta/est_cabezas.por_linea, alineada con las líneas): [{palabras:
    [{palabra, negrita, cursiva, versalita, ...}], inicio}] o pagina["estilos"]. Con estilos, el inicio de entrada aprendido
    y el lema en negrita o versalita mandan; la categoría y los ejemplos salen de la cursiva.
    Devuelve [{lema, var, gram, etim, acepciones: [{n, def, ejemplos}], remite, texto, lineas: [i...]}]."""
    estilos = estilos if estilos is not None else pagina.get("estilos")
    ls = pagina.get("lineas", [])
    zonas = pagina.get("zonas", [])
    cab = {"RunningTitleZone", "NumberingZone", "Page-header", "Page-footer", "QuireMarksZone"}
    cuerpo = [i for i, l in enumerate(ls) if l.get("texto", "").strip() and
              (clase(zonas[l["zona"]]) if 0 <= l.get("zona", -1) < len(zonas) else "MainZone") not in cab]
    if not cuerpo:
        return []
    # columnas: por zona (o por agrupación de x0 si no hay zona)
    grupos = {}
    for i in cuerpo:
        grupos.setdefault(ls[i].get("zona", -1), []).append(i)
    inicio = set()
    for z, idx in grupos.items():
        x0s = sorted(ls[i]["caja"][0] for i in idx)
        alto = sorted(ls[i]["caja"][3] - ls[i]["caja"][1] for i in idx)[len(idx) // 2] or 10
        izq = x0s[max(0, int(len(x0s) * 0.05))]
        fuera = [i for i in idx if ls[i]["caja"][0] - izq < 0.5 * alto]
        dentro = [i for i in idx if ls[i]["caja"][0] - izq >= 0.5 * alto]
        francesa = len(dentro) >= len(fuera)  # la mayoría entra: las que salen abren entrada
        for i in idx:
            sale = ls[i]["caja"][0] - izq < 0.5 * alto
            geo = (sale if francesa else not sale) and len(dentro) > 0 and len(fuera) > 0
            if estilos and i < len(estilos) and estilos[i].get("inicio") is not None:
                pi = estilos[i]["inicio"]
                pal = estilos[i].get("palabras") or []
                marcado = bool(pal) and max(pal[0].get("negrita", 0), pal[0].get("versalita", 0)) >= umbral
                if pi >= umbral or (pi >= 0.3 and (marcado or _parece_inicio(ls[i]["texto"], geo))):
                    inicio.add(i)
            elif _parece_inicio(ls[i]["texto"], geo):
                inicio.add(i)
    entradas, cur = [], None
    for i in cuerpo:
        if i in inicio or cur is None:
            cur = {"lineas": [], "texto": ""}
            entradas.append(cur)
        cur["lineas"].append(i)
        if estilos and i < len(estilos):
            cur.setdefault("fichas", []).extend(estilos[i].get("palabras") or [])
        t = ls[i]["texto"].strip()
        cur["texto"] = (cur["texto"][:-1] + t) if cur["texto"].endswith("-") else (cur["texto"] + " " + t).strip()
    return [analizar_entrada(e, umbral) for e in entradas]


def _con_estilos(out, fichas, umbral):
    """Corrige lema, categoría y ejemplos con el estilo por palabra (negrita/versalita = lema; cursiva = categoría o ejemplo)."""
    marca = lambda f: max(f.get("negrita", 0), f.get("versalita", 0)) >= umbral
    cur = lambda f: f.get("cursiva", 0) >= umbral
    k = 0
    while k < len(fichas) and k < 5 and marca(fichas[k]):
        k += 1
    if k:
        lema = " ".join(f["palabra"] for f in fichas[:k]).strip(" ,.;:")
        if lema and (not out["lema"] or _cer(lema, out["lema"]) > 0.3):
            out["lema"] = lema
    j = k
    while j < len(fichas) and not cur(fichas[j]) and j < k + 6:
        j += 1
    g = []
    while j < len(fichas) and cur(fichas[j]) and len(g) < 4 and len(fichas[j]["palabra"]) <= 8:
        g.append(fichas[j]["palabra"])
        j += 1
    if g and not out["gram"] and RE_GRAM.match(" ".join(g)):
        out["gram"] = " ".join(g)
    # ejemplos: tramos en cursiva de 3 o más palabras dentro de las acepciones
    tramos, act = [], []
    for f in fichas[j:]:
        if cur(f):
            act.append(f["palabra"])
        else:
            if len(act) >= 3:
                tramos.append(" ".join(act))
            act = []
    if len(act) >= 3:
        tramos.append(" ".join(act))
    for t in tramos:
        for a in out["acepciones"]:
            if t in a["def"]:
                a["def"] = a["def"].replace(t, "").strip()
                a["ejemplos"].append(t)
                break
    return out


def analizar_entrada(e, umbral=0.5):
    out = _analizar(e)
    if e.get("fichas"):
        out = _con_estilos(out, e["fichas"], umbral)
    return out


def _analizar(e):
    t = e["texto"]
    out = {"lineas": e["lineas"], "texto": t, "lema": "", "var": "", "gram": "", "etim": "", "acepciones": [], "remite": ""}
    m = RE_LEMA.match(t)
    if not m:
        out["acepciones"] = [{"n": None, "def": t, "ejemplos": []}]
        return out
    out["lema"] = m.group(1).lstrip("☞ ").strip()
    resto = t[m.end():]
    mv = re.match(r"^\s*(-?[a-záéíóúñç]{1,4}),\s", resto)  # variante de género: «blanco, ca,»
    if mv:
        out["var"] = mv.group(1)
        resto = resto[mv.end():]
    me = RE_ETIM.match(resto)
    if me:
        out["etim"] = me.group(0).strip()
        resto = resto[me.end():]
    mg = RE_GRAM.match(resto)
    if mg:
        out["gram"] = mg.group(1).strip()
        resto = resto[mg.end():]
    mr = RE_REMITE.match(resto)
    if mr and len(resto) < 60:
        out["remite"] = resto[mr.end():].strip(" .")
        return out
    partes = RE_NUM.split(resto)
    # split con grupo: [antes, n1, texto1, n2, texto2...]; «||» da None como número
    acs = []
    if len(partes) > 1:
        if partes[0].strip():
            acs.append({"n": None, "def": partes[0].strip()})
        for k in range(1, len(partes) - 1, 2):
            acs.append({"n": partes[k], "def": (partes[k + 1] or "").strip()})
    else:
        acs.append({"n": None, "def": resto.strip()})
    for a in acs:  # ejemplo: tras la definición, frase que empieza en mayúscula después de punto (heurística débil)
        a["ejemplos"] = []
    out["acepciones"] = [a for a in acs if a["def"]]
    return out


def detectar_tipo(pagina):
    """«diccionario», «comic» o «texto» (las figuras de iluminados y álbumes ya las trata la salida general)."""
    ls, zonas = pagina.get("lineas", []), pagina.get("zonas", [])
    figs = [z["caja"] for z in zonas if clase(z) in ("Picture", "GraphicZone") and z.get("conf", 1) >= 0.3]
    con_texto = [l for l in ls if l.get("texto", "").strip()]
    if len(figs) >= 3 and con_texto:
        dentro = sum(1 for l in con_texto if any(l["caja"][0] >= f[0] - 4 and l["caja"][1] >= f[1] - 4 and l["caja"][2] <= f[2] + 4 and l["caja"][3] <= f[3] + 4 for f in figs))
        if dentro >= 0.4 * len(con_texto):
            return "comic"
    es = [e for e in entradas_diccionario(pagina) if e["lema"]]
    fuertes = sum(1 for e in es if e["gram"] or e["remite"] or len([c for c in e["lema"] if c.isalpha()]) >= 3 and e["lema"].isupper())
    if len(es) >= 4 and fuertes >= max(3, 0.4 * len(es)):
        return "diccionario"
    return "texto"


# ---------------------------------------------------------------- estilo por palabra en las salidas
REND = {"negrita": "bold", "cursiva": "italic", "versalita": "smallcaps", "subrayado": "underline", "color": "color(red)"}


def rasgos(palabra, umbral=0.5):
    """Lista de rasgos activos de una palabra de la cabeza tipográfica ({palabra, negrita, cursiva, ...})."""
    return [k for k in REND if palabra.get(k, 0) >= umbral]


def _tramos(texto, palabras, umbral):
    """Parte el texto de la línea en tramos (texto, rasgos) agrupando palabras seguidas con los mismos rasgos."""
    if not palabras:
        return [(texto, [])]
    out, pos = [], 0
    for w in palabras:
        a, b = w.get("ini", pos), w.get("fin", pos + len(w.get("palabra", "")))
        if a > pos:
            out.append((texto[pos:a], None))  # espacios entre palabras: heredan
        out.append((texto[a:b], rasgos(w, umbral)))
        pos = b
    if pos < len(texto):
        out.append((texto[pos:], []))
    # los huecos entre palabras toman los rasgos comunes a sus dos vecinas; después se fusiona
    for i, (t, r) in enumerate(out):
        if r is None:
            a_ = out[i - 1][1] if i > 0 and out[i - 1][1] is not None else []
            b_ = out[i + 1][1] if i + 1 < len(out) and out[i + 1][1] is not None else []
            out[i] = (t, [k for k in a_ if k in b_])
    fus = []
    for t, r in out:
        if fus and fus[-1][1] == r:
            fus[-1] = (fus[-1][0] + t, r)
        else:
            fus.append((t, r))
    return fus


def linea_tei(texto, palabras, umbral=0.5):
    """Texto de la línea con <hi rend="bold italic smallcaps underline color(red)"> donde la cabeza lo ve."""
    o = []
    for t, r in _tramos(texto, palabras, umbral):
        if r and t.strip():
            lead = t[: len(t) - len(t.lstrip())]
            tail = t[len(t.rstrip()):]
            o.append(f'{escape(lead)}<hi rend="{" ".join(REND[k] for k in r)}">{escape(t.strip())}</hi>{escape(tail)}')
        else:
            o.append(escape(t))
    return "".join(o)


def linea_html(texto, palabras, umbral=0.5):
    o = []
    for t, r in _tramos(texto, palabras, umbral):
        x = escape(t)
        if r and t.strip():
            if "negrita" in r:
                x = f"<b>{x}</b>"
            if "cursiva" in r:
                x = f"<i>{x}</i>"
            if "subrayado" in r:
                x = f"<u>{x}</u>"
            st = ";".join(s_ for k, s_ in (("versalita", "font-variant:small-caps"), ("color", "color:#a01818")) if k in r)
            if st:
                x = f'<span style="{st}">{x}</span>'
        o.append(x)
    return "".join(o)


def linea_markdown(texto, palabras, umbral=0.5):
    o = []
    for t, r in _tramos(texto, palabras, umbral):
        if r and t.strip():
            lead, core, tail = t[: len(t) - len(t.lstrip())], t.strip(), t[len(t.rstrip()):]
            m = ("**" if "negrita" in r else "") + ("*" if "cursiva" in r else "")
            if "subrayado" in r:
                core = f"<u>{core}</u>"
            o.append(f"{lead}{m}{core}{m[::-1]}{tail}")
        else:
            o.append(t)
    return "".join(o)


def alto_estilos(palabras, umbral=0.5):
    """Para ALTO: lista de (palabra, FONTSTYLE) con FONTSTYLE de ALTO (bold, italics, smallcaps, underline) o None, y
    los TextStyle que hacen falta: {FONTSTYLE: id}. El color va en FONTCOLOR (rojo de rúbrica: «A01818»)."""
    mapa = {"negrita": "bold", "cursiva": "italics", "versalita": "smallcaps", "subrayado": "underline"}
    out, estilos = [], {}
    for w in palabras:
        r = rasgos(w, umbral)
        fs = " ".join(mapa[k] for k in r if k in mapa) or None
        col = "A01818" if "color" in r else None
        clave = (fs, col)
        if fs or col:
            estilos.setdefault(clave, f"TS_{len(estilos) + 1}")
        out.append((w.get("palabra", ""), estilos.get(clave)))
    return out, {v: {"FONTSTYLE": k[0], "FONTCOLOR": k[1]} for k, v in estilos.items()}


# ---------------------------------------------------------------- salidas
def _orden_zonas(pagina):
    return pagina.get("zonas", [])


def a_tei(pagina, tipo=None, id_pag="p1"):
    tipo = tipo or pagina.get("estructura") or "texto"
    ls, zonas = pagina.get("lineas", []), pagina.get("zonas", [])
    o = [f'<div type="page" xml:id="{id_pag}">']
    est = pagina.get("estilos")
    tx = lambda i: linea_tei(ls[i]["texto"], est[i].get("palabras") or []) if est and i < len(est) else escape(ls[i]["texto"])
    if tipo == "diccionario":
        for e in entradas_diccionario(pagina):
            if not e["lema"]:
                o.append(f'  <p>{escape(e["texto"])}</p>')
                continue
            o.append('  <entry>')
            o.append(f'    <form type="lemma"><orth>{escape(e["lema"])}</orth>' + (f'<orth type="variant">{escape(e["var"])}</orth>' if e["var"] else "") + '</form>')
            if e["etim"]:
                o.append(f'    <etym>{escape(e["etim"])}</etym>')
            if e["gram"]:
                o.append(f'    <gramGrp><gram type="pos">{escape(e["gram"])}</gram></gramGrp>')
            if e["remite"]:
                o.append(f'    <xr type="see"><ref>{escape(e["remite"])}</ref></xr>')
            for a in e["acepciones"]:
                n = f' n="{escape(a["n"])}"' if a["n"] else ""
                o.append(f'    <sense{n}><def>{escape(a["def"])}</def>' + "".join(f'<cit type="example"><quote>{escape(x)}</quote></cit>' for x in a["ejemplos"]) + '</sense>')
            o.append('  </entry>')
    else:
        # figuras enlazadas con el texto más cercano (iluminados, álbumes) y viñetas con sus bocadillos (cómics)
        figs = [k for k, z in enumerate(zonas) if clase(z) in ("GraphicZone", "Picture")]
        por_zona = {}
        for i, l in enumerate(ls):
            por_zona.setdefault(l.get("zona", -1), []).append(i)
        dentro = lambda c, z: c[0] >= z[0] - 4 and c[1] >= z[1] - 4 and c[2] <= z[2] + 4 and c[3] <= z[3] + 4
        if tipo == "comic":
            vin = sorted(figs, key=lambda k: min([i for i, l in enumerate(ls) if dentro(l["caja"], zonas[k]["caja"])] or [10**9]))
            usadas = set()
            for n, k in enumerate(vin, 1):
                o.append(f'  <div type="panel" n="{n}" facs="#{id_pag}_z{k}">')
                for i, l in enumerate(ls):
                    if i not in usadas and dentro(l["caja"], zonas[k]["caja"]) and l.get("texto", "").strip():
                        usadas.add(i)
                        o.append(f'    <said><l>{tx(i)}</l></said>')
                o.append('  </div>')
            resto = [i for i in range(len(ls)) if i not in usadas and ls[i].get("texto", "").strip()]
            if resto:
                o.append('  <div type="text">' + "".join(f'<l>{tx(i)}</l>' for i in resto) + '</div>')
        else:
            for k in figs:
                z = zonas[k]["caja"]
                # texto vecino: la línea más cercana que no esté dentro (para @corresp)
                cerca = min(range(len(ls)), key=lambda i: abs((ls[i]["caja"][1] + ls[i]["caja"][3]) / 2 - (z[1] + z[3]) / 2), default=None) if ls else None
                interior = [ls[i]["texto"] for i in range(len(ls)) if dentro(ls[i]["caja"], z) and ls[i].get("texto", "").strip()]
                o.append(f'  <figure xml:id="{id_pag}_fig{k}" facs="#{id_pag}_z{k}"' + (f' corresp="#{id_pag}_l{cerca}"' if cerca is not None else "") + '>' +
                         "".join(f'<label>{escape(t)}</label>' for t in interior) + '</figure>')
            for i, l in enumerate(ls):
                if l.get("texto", "").strip() and not any(dentro(l["caja"], zonas[k]["caja"]) for k in figs):
                    rol = clase(zonas[l["zona"]]) if 0 <= l.get("zona", -1) < len(zonas) else ""
                    etq = {"HeadingLine": "head"}.get(l.get("tipo", ""), "l")
                    o.append(f'  <{etq} xml:id="{id_pag}_l{i}"' + (f' type="{rol}"' if rol else "") + f'>{tx(i)}</{etq}>')
    o.append('</div>')
    return "\n".join(o)


def a_json(pagina, tipo=None):
    tipo = tipo or pagina.get("estructura") or "texto"
    if tipo == "diccionario":
        return {"tipo": tipo, "entradas": [{k: v for k, v in e.items()} for e in entradas_diccionario(pagina)]}
    return {"tipo": tipo, "zonas": pagina.get("zonas", []), "lineas": [{k: l.get(k) for k in ("texto", "caja", "zona")} for l in pagina.get("lineas", [])]}


def a_markdown(pagina, tipo=None):
    tipo = tipo or pagina.get("estructura") or "texto"
    if tipo == "diccionario":
        o = []
        for e in entradas_diccionario(pagina):
            if not e["lema"]:
                o.append(e["texto"])
                continue
            s = f'**{e["lema"]}**' + (f', {e["var"]}' if e["var"] else "") + (f' {e["etim"]}' if e["etim"] else "") + (f' *{e["gram"]}*' if e["gram"] else "")
            if e["remite"]:
                s += f' → {e["remite"]}'
            for a in e["acepciones"]:
                s += (f' **{a["n"]}.** ' if a["n"] else " ") + a["def"]
            o.append(s)
        return "\n\n".join(o)
    return pagina.get("texto", "")
