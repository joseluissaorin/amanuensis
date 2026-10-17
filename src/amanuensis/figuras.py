"""Extracción estructurada de imágenes: figuras, grabados, capitulares, sellos y fórmulas.

A partir de las zonas de imagen de la maquetación (con su vector en el espacio común de EmbeddingGemma 2 y
descriptores Iconclass recuperados, no generados):
- recorte a la resolución original (en PDF, la página se vuelve a renderizar a 300 ppp);
- nombres estables img/<documento>/pNNN_figMM.webp, en orden de lectura (de arriba abajo);
- deduplicado de ornamentos repetidos por vector (coseno > 0,95 con otro del mismo tipo);
- enlaces: figura-pie (línea que empieza por «Fig.», «Figura», «Lámina», «Table»… junto a la figura, o la zona
  Caption de la maqueta), llamadas en el texto («fig. 3») y capitular-palabra (primera palabra de la línea que
  la capitular encabeza);
- texto alternativo: el pie si lo hay; si no, el tipo y el primer descriptor Iconclass.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image

NOMBRE_TIPO = {"figura": "Figura", "ilustracion": "Ilustración", "capitular": "Capitular", "sello_tinta": "Sello de tinta",
               "sello": "Sello", "formula": "Fórmula"}
PIE = re.compile(r"^\s*(fig(?:ura|ure|\.)?|l[áa]m(?:ina|\.)?|plate|grabado|tabla|table|imagen|image|il+ustraci[oó]n|illustration)\s*\.?\s*([0-9]+|[IVXLC]+)\b", re.I)
LLAMADA = re.compile(r"\b(?:fig(?:ura|ure)?s?|l[áa]m(?:ina)?s?)\.?\s*([0-9]+)\b", re.I)
DUP = 0.95


def _hmed(lineas):
    hs = sorted(l.caja[3] - l.caja[1] for l in lineas if l.texto.strip())
    return hs[len(hs) // 2] if hs else 20.0


def _solape_x(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(1.0, min(a[2] - a[0], b[2] - b[0]))


def _dentro(l, c, m=0.0):
    cx, cy = (l.caja[0] + l.caja[2]) / 2, (l.caja[1] + l.caja[3]) / 2
    return c[0] - m <= cx <= c[2] + m and c[1] - m <= cy <= c[3] + m


def procesar(paginas, figs_mq, originales, stem: str, escala=None):
    """paginas: [Pagina]; figs_mq: por página, lista de figuras de la maqueta (o None); originales: por página,
    función que devuelve la imagen en color a resolución original y el factor caja->píxel. Devuelve
    {ruta relativa: Image} con los recortes a guardar."""
    imagenes, vistos = {}, []  # vistos: (vector, tipo, id, fichero)
    for k, (p, fm) in enumerate(zip(paginas, figs_mq)):
        p.figuras = []
        if not fm:
            continue
        hm = _hmed(p.lineas)
        orig, f = originales[k]()
        for j, z in enumerate(fm, 1):
            fid = f"p{k + 1:03d}_fig{j:02d}"
            c = [float(v) for v in z["caja"]]
            fig = {"id": fid, "tipo": z["tipo"], "caja": [round(v, 1) for v in c], "conf": round(float(z["conf"]), 3),
                   "fichero": f"img/{stem}/{fid}.webp", "pie": "", "pie_lineas": [], "llamadas": [], "palabra": None,
                   "iconclass": z.get("iconclass", []), "duplicado_de": None}
            vec = np.array(z.get("vector") or [], np.float32)
            for v2, t2, id2, fich2 in vistos:
                if len(vec) and t2 == z["tipo"] and float(vec @ v2) > DUP:
                    fig["duplicado_de"], fig["fichero"] = id2, fich2
                    break
            if fig["duplicado_de"] is None:
                m = 0.03 * max(c[2] - c[0], c[3] - c[1])  # margen: las cajas de la maqueta salen algo justas
                x0, y0, x1, y1 = (int(round(v * f)) for v in (c[0] - m, c[1] - m, c[2] + m, c[3] + m))
                imagenes[fig["fichero"]] = orig.crop((max(0, x0), max(0, y0), min(orig.size[0], x1), min(orig.size[1], y1)))
                if len(vec):
                    vistos.append((vec, z["tipo"], fid, fig["fichero"]))
            # líneas dentro de la figura (rótulos de un diagrama, texto de un sello): no son texto corrido
            for i, l in enumerate(p.lineas):
                if _dentro(l, c) and z["tipo"] != "capitular":
                    l.en_figura = fid
            if z["tipo"] == "capitular":
                cand = [(l.caja[1], i) for i, l in enumerate(p.lineas)
                        if l.caja[0] >= c[0] + 0.5 * (c[2] - c[0]) and c[1] - 0.5 * hm <= (l.caja[1] + l.caja[3]) / 2 <= c[3]
                        and l.caja[0] - c[2] < 3 * hm and l.texto.strip()]
                if cand:
                    i = min(cand)[1]
                    fig["palabra"] = {"linea": i, "texto": p.lineas[i].texto.split()[0]}
            else:
                fig["pie_lineas"], fig["pie"] = _pie(p, c, hm, z)
            fig["alt"] = fig["pie"] or _alt(fig)
            p.figuras.append(fig)
    # llamadas «fig. N» en el texto -> figura cuyo pie lleva el número N
    numeros = {}
    for k, p in enumerate(paginas):
        for fig in getattr(p, "figuras", []):
            m = PIE.match(fig["pie"] or "")
            if m:
                numeros.setdefault(m.group(2).upper(), fig)
    for k, p in enumerate(paginas):
        pies = {i for fig in getattr(p, "figuras", []) for i in fig["pie_lineas"]}
        for i, l in enumerate(p.lineas):
            if i in pies or getattr(l, "en_figura", None):
                continue
            for m in LLAMADA.finditer(l.texto):
                fig = numeros.get(m.group(1))
                if fig is not None:
                    fig["llamadas"].append({"pagina": k + 1, "linea": i, "texto": m.group(0)})
    return imagenes


def _pie(p, c, hm, z):
    """Líneas del pie: zona Caption de la maqueta que toca la figura o línea «Fig. N…» junto a ella."""
    lineas = p.lineas
    ws = max(1.0, c[2] - c[0])
    cand = []
    for i, l in enumerate(lineas):
        if getattr(l, "en_figura", None) or not l.texto.strip():
            continue
        debajo = 0 <= l.caja[1] - c[3] <= 4 * hm
        encima = 0 <= c[1] - l.caja[3] <= 4 * hm
        if (debajo or encima) and _solape_x(l.caja, c) > 0.3 and (PIE.match(l.texto) or getattr(l, "zona_clase", "") == "Caption"):
            cand.append((0 if debajo else 1, abs(l.caja[1] - c[3]) if debajo else abs(c[1] - l.caja[3]), i))
    if not cand:
        return [], ""
    _, _, i = min(cand)
    sel = [i]
    for j in range(i + 1, min(len(lineas), i + 6)):  # el pie sigue mientras las líneas vengan pegadas y alineadas
        a, b = lineas[sel[-1]], lineas[j]
        if 0 <= b.caja[1] - a.caja[3] < 0.8 * hm and _solape_x(a.caja, b.caja) > 0.3 and not PIE.match(b.texto) and abs(b.caja[1] - a.caja[1]) < 2.5 * hm:
            sel.append(j)
        else:
            break
    texto = " ".join(lineas[j].texto.strip() for j in sel)
    for j in sel:
        lineas[j].pie_de = True
    return sel, texto


def _alt(fig):
    t = NOMBRE_TIPO.get(fig["tipo"], "Imagen")
    if fig["iconclass"]:
        ic = fig["iconclass"][0]
        return f"{t}: {ic.get('es') or ic['en']} (Iconclass {ic['notacion']}, sugerencia por similitud)"
    return t
