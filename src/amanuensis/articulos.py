"""Artículos de una página de prensa (modo prensa), por reglas sobre los bloques en orden de lectura (de maqueta/articulos.py,
medido en GRESEL: B-cubed F1 84,8 frente a 68,7 de «página entera»). Un artículo empieza en cada titular (bloque de cuerpo
>= ALTO veces el del texto corriente, o corto y en mayúsculas) que sigue a un bloque de texto; los titulares seguidos
(antetítulo, título, subtítulos) van juntos en la cabeza del artículo. Se calcula al exportar (los índices de bloque son los
de `salidas.bloques(p)` de la página ya pulida): `paginas[].articulos` en JSON, separador en Markdown y `<div type="article">` en TEI.
"""
from __future__ import annotations

import numpy as np

def _mayus(t: str) -> bool:
    letras = [c for c in t if c.isalpha()]
    return len(letras) >= 4 and sum(c.isupper() for c in letras) >= 0.8 * len(letras)


REGLAS_V1 = {"alto": 1.35, "max_lineas": 0, "ancho_min": 0.0, "mayus": True, "sig_cuerpo": False, "max_car": 0, "clase": 0}
REGLAS_V2 = {"alto": 1.25, "max_lineas": 0, "ancho_min": 0.0, "mayus": False, "sig_cuerpo": True, "max_car": 0, "clase": 0}
REGLAS = REGLAS_V1  # maqueta (11-11): contra la verdad plata las v2 no ganan (79,35 frente a 79,76 de B-cubed); v2: REGLAS_V2


def titulares(pag: dict, alto: float | None = None, **reglas) -> list[bool]:
    """Bloques que abren artículo. Reglas (REGLAS por defecto):
      alto        cuerpo mediano del bloque ≥ alto × el del texto corriente
      mayus       o bloque corto (≤ 3 renglones, ≤ 120 car.) en mayúsculas y algo mayor que el texto (≥ 1,05)
      max_lineas  >0: un titular no pasa de N renglones (los bloques grandes de cuerpo grueso, anuncios y cabeceras, no abren)
      max_car     >0: ni de N caracteres
      ancho_min   el bloque mide al menos esa fracción del ancho de columna (mediana de los renglones de cuerpo): los
                  filetes, firmas y números sueltos no abren
      sig_cuerpo  solo abre si detrás (en el orden) viene un bloque que no es titular de cuerpo corriente
      clase       usa la clase de zona de la maqueta (zona_clase de los renglones): 0 no; 1 titular también si la mayoría es
                  Section-header/Title; 2 SOLO si lo es; con 1 y 2, las cabeceras (Page-header, RunningTitleZone) nunca abren"""
    r = dict(REGLAS, **reglas)
    if alto is not None:
        r["alto"] = alto
    ls = pag["lineas"]
    hs = np.array([l["caja"][3] - l["caja"][1] for l in ls], float) if ls else np.zeros(0)
    if not len(hs):
        return []
    h = float(np.median(hs))
    ws = np.array([l["caja"][2] - l["caja"][0] for l in ls], float)
    col = float(np.median(ws[hs <= 1.2 * h])) if (hs <= 1.2 * h).any() else float(np.median(ws))
    out, cuerpo = [], []
    for b in pag.get("bloques", []):
        ii = [i for i in b["lineas"] if 0 <= i < len(ls)]
        if not ii:
            out.append(False); cuerpo.append(False)
            continue
        hb = float(np.median(hs[ii]))
        t = " ".join(ls[i]["texto"] for i in ii)
        anchob = max(ls[i]["caja"][2] for i in ii) - min(ls[i]["caja"][0] for i in ii)
        es = hb >= r["alto"] * h or (r["mayus"] and len(ii) <= 3 and len(t) <= 120 and _mayus(t) and hb >= 1.05 * h)
        if r.get("clase"):
            cs = [ls[i].get("zona_clase") for i in ii]
            mayor = max(set(cs), key=cs.count)
            tit_c = mayor in ("Section-header", "Title")
            es = tit_c if r["clase"] == 2 else (es or tit_c)
            if mayor in ("Page-header", "RunningTitleZone", "Page-footer", "NumberingZone"):
                es = False
        if es and r["max_lineas"] and len(ii) > r["max_lineas"]:
            es = False
        if es and r["max_car"] and len(t) > r["max_car"]:
            es = False
        if es and r["ancho_min"] and anchob < r["ancho_min"] * col:
            es = False
        out.append(es)
        cuerpo.append(hb < 1.2 * h and len(ii) >= 2)
    if r["sig_cuerpo"]:
        for k in range(len(out)):
            if out[k]:
                j = k + 1
                while j < len(out) and out[j]:
                    j += 1
                if j >= len(out) or not cuerpo[j]:
                    out[k] = False
    return out


def articulos(pag: dict, alto: float | None = None, **reglas) -> list[dict]:
    bl = pag.get("bloques", [])
    tit = titulares(pag, alto, **reglas)
    arts, act, previo_tit = [], None, False
    for k, (b, t) in enumerate(zip(bl, tit)):
        if act is None or (t and not previo_tit):
            act = {"bloques": [], "titulo": ""}
            arts.append(act)
        act["bloques"].append(k)
        if t and (previo_tit or len(act["bloques"]) == 1):
            act["titulo"] = (act["titulo"] + " " + b.get("texto", "")).strip()
        previo_tit = t
    return arts


def de_pagina(p) -> list[dict]:
    """[{"bloques": [índices en salidas.bloques(p)], "titulo": str}] si la página es de prensa; si no, []. Con `--articulos red` (p._art_red,
    grupos de renglones calculados al leer) los bloques de la página ya pulida se asignan al artículo de su primer renglón; si no, reglas v1."""
    if "prensa" not in (getattr(p, "fuente", "") or ""):
        return []
    from .salidas import bloques, unir

    bs = [b for b in bloques(p)]
    if len(bs) < 2:
        return []
    red = getattr(p, "_art_red", None)
    if red:
        dueno = {id(l): k for k, a in enumerate(red) for l in a["lineas"]}
        etiq, ult = [], 0
        for b in bs:
            k = next((dueno[id(p.lineas[i])] for i in b["lineas"] if id(p.lineas[i]) in dueno), None)
            ult = ult if k is None else k
            etiq.append(ult)
        out = []
        for k in sorted(set(etiq), key=etiq.index):
            idx = [j for j, e in enumerate(etiq) if e == k]
            out.append({"bloques": idx, "titulo": red[k].get("titulo", ""), "confianza": red[k].get("confianza", 1.0)})
        return out
    pag = {"lineas": [{"caja": l.caja, "texto": l.texto} for l in p.lineas],
           "bloques": [{"lineas": b["lineas"], "texto": b.get("texto") or unir([p.lineas[i] for i in b["lineas"]])} for b in bs]}
    return articulos(pag)
