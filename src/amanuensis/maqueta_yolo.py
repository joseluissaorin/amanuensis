"""Maquetación por regiones con DocLayout-YOLO (opcional): bloques de texto, figuras, pies, cabeceras y folios.

DocLayout-YOLO-N (opendatalab, release-v2, 2,2 M parámetros) afinado en D4LA, exportado a ONNX una vez fuera del
paquete; aquí solo se ejecuta con onnxruntime (sin el código de Ultralytics, que es AGPL). La licencia de los pesos no
está declarada («unknown» en su ficha): por eso no está en el catálogo ni se descarga sola, y solo se usa si se pide
(AMANUENSIS_MAQUETA_YOLO_SI=1, o AMANUENSIS_MAQUETA_YOLO con la carpeta) y está instalado en
`~/.cache/amanuensis/doclayout-yolo-n-d4la/` (modelo.onnx + clases.json) o en AMANUENSIS_MAQUETA_YOLO. Tenerlo en la
caché no basta: así la lectura es la misma que la de la demo, Yappy y la app, que no lo llevan.

Lo que aporta a Ligero en prensa y revistas: el detector de renglones no sabe de columnas ni de fotos; las regiones sí.
Cada renglón va a su región; las regiones de texto se ordenan con un corte XY en el que no cuentan figuras ni pies (son
los que tapan los canales: una foto encima de dos columnas), los renglones de dentro de una figura quedan fuera de la
lectura, los pies al final y cabeceras y folios como mobiliario.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
from PIL import Image

LADO = 1600
CONF = 0.25
IOU = 0.5

# clases de D4LA -> clase de zona de Amanuensis (salidas.ROL); None = texto corrido
CLASE = {"DocTitle": "Section-header", "ParaTitle": "Section-header", "RegionTitle": "Section-header",
         "Figure": "GraphicZone", "FigureName": "Caption", "TableName": "Caption", "PageHeader": "RunningTitleZone",
         "PageFooter": "Page-footer", "Footer": "Page-footer", "Number": "NumberingZone", "PageNumber": "NumberingZone",
         "Table": None, "Equation": None}
FUERA_DEL_CORTE = {"Figure", "FigureName", "TableName", "PageHeader", "PageFooter", "Footer", "Number", "PageNumber"}


def carpeta() -> str | None:
    for r in (os.environ.get("AMANUENSIS_MAQUETA_YOLO"),
              os.path.join(os.environ.get("AMANUENSIS_CACHE") or os.path.expanduser("~/.cache/amanuensis"), "doclayout-yolo-n-d4la")):
        if r and os.path.exists(os.path.join(r, "modelo.onnx")):
            return r
    return None


def activa() -> str | None:
    """Carpeta del modelo si se ha pedido (AMANUENSIS_MAQUETA_YOLO_SI=1 o AMANUENSIS_MAQUETA_YOLO) y está instalado."""
    if os.environ.get("AMANUENSIS_MAQUETA_YOLO_NO") == "1":
        return None
    if os.environ.get("AMANUENSIS_MAQUETA_YOLO_SI") != "1" and not os.environ.get("AMANUENSIS_MAQUETA_YOLO"):
        return None
    return carpeta()


@lru_cache(maxsize=1)
def modelo(ruta: str):
    import onnxruntime as ort

    s = ort.InferenceSession(os.path.join(ruta, "modelo.onnx"), providers=["CPUExecutionProvider"])
    nombres = json.load(open(os.path.join(ruta, "clases.json"), encoding="utf-8"))
    return s, {int(k): v for k, v in nombres.items()}


def _nms(cajas: np.ndarray, conf: np.ndarray, iou: float) -> list[int]:
    orden = conf.argsort()[::-1]
    out = []
    while len(orden):
        i = orden[0]
        out.append(int(i))
        if len(orden) == 1:
            break
        r = cajas[orden[1:]]
        x0 = np.maximum(cajas[i, 0], r[:, 0]); y0 = np.maximum(cajas[i, 1], r[:, 1])
        x1 = np.minimum(cajas[i, 2], r[:, 2]); y1 = np.minimum(cajas[i, 3], r[:, 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        a = (cajas[i, 2] - cajas[i, 0]) * (cajas[i, 3] - cajas[i, 1])
        b = (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
        orden = orden[1:][inter / np.maximum(a + b - inter, 1e-6) < iou]
    return out


def regiones(img: Image.Image, ruta: str | None = None) -> list[dict]:
    """[{clase, caja [x0, y0, x1, y1] en px de la página, conf}] de la página."""
    ruta = ruta or carpeta()
    if ruta is None:
        return []
    s, nombres = modelo(ruta)
    rgb = img.convert("RGB")
    w, h = rgb.size
    esc = LADO / max(w, h)
    nw, nh = int(round(w * esc)), int(round(h * esc))
    lienzo = Image.new("RGB", (LADO, LADO), (114, 114, 114))
    dx, dy = (LADO - nw) // 2, (LADO - nh) // 2
    lienzo.paste(rgb.resize((nw, nh), Image.BILINEAR), (dx, dy))
    x = np.asarray(lienzo, np.float32).transpose(2, 0, 1)[None] / 255.0
    out = s.run(None, {s.get_inputs()[0].name: x})[0][0]  # [4 + clases, propuestas]
    cajas, puntos = out[:4].T, out[4:].T
    cls = puntos.argmax(1)
    conf = puntos[np.arange(len(cls)), cls]
    ok = conf >= CONF
    cajas, cls, conf = cajas[ok], cls[ok], conf[ok]
    xyxy = np.stack([cajas[:, 0] - cajas[:, 2] / 2, cajas[:, 1] - cajas[:, 3] / 2,
                     cajas[:, 0] + cajas[:, 2] / 2, cajas[:, 1] + cajas[:, 3] / 2], 1)
    res = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        for k in _nms(xyxy[idx], conf[idx], IOU):
            b = xyxy[idx[k]]
            caja = [float((b[0] - dx) / esc), float((b[1] - dy) / esc), float((b[2] - dx) / esc), float((b[3] - dy) / esc)]
            caja = [max(0.0, caja[0]), max(0.0, caja[1]), min(float(w), caja[2]), min(float(h), caja[3])]
            res.append({"clase": nombres[int(c)], "caja": caja, "conf": float(conf[idx[k]])})
    return res


def _corte_xy(cajas: list, idx: list[int]) -> list[int]:
    """Orden de lectura de regiones: corte vertical (columnas, de izquierda a derecha) y, si no lo hay, horizontal
    (de arriba abajo), recursivo; sin corte posible, por su esquina superior."""
    if len(idx) <= 1:
        return idx
    for eje in (0, 1):  # 0: columnas (x); 1: franjas (y)
        a0, a1 = (0, 2) if eje == 0 else (1, 3)
        orden = sorted(idx, key=lambda i: cajas[i][a0])
        grupos, act, fin = [], [orden[0]], cajas[orden[0]][a1]
        for i in orden[1:]:
            if cajas[i][a0] >= fin - 2:
                grupos.append(act)
                act = [i]
            else:
                act.append(i)
            fin = max(fin, cajas[i][a1])
        grupos.append(act)
        if len(grupos) > 1:
            return [j for g in grupos for j in _corte_xy(cajas, g)]
    return sorted(idx, key=lambda i: (cajas[i][1], cajas[i][0]))


def clasificar(img: Image.Image, lineas: list) -> tuple[list, dict]:
    """Solo las clases: (clase de zona de cada renglón o None, resumen). El orden lo pone otro (columnas o el aprendido)."""
    regs = regiones(img)
    if not regs or not lineas:
        return [None] * len(lineas), {}

    def solape(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    clases, en_fig = [], 0
    for c in lineas:
        area = max(1.0, (c[2] - c[0]) * (c[3] - c[1]))
        r = max(regs, key=lambda g: solape(c, g["caja"]))
        cl = CLASE.get(r["clase"]) if solape(c, r["caja"]) >= 0.5 * area else None
        en_fig += cl == "GraphicZone"
        clases.append(cl)
    return clases, {"regiones": len(regs), "figuras": sum(r["clase"] == "Figure" for r in regs), "renglones_en_figuras": en_fig}


def maquetar(img: Image.Image, lineas: list) -> tuple[list, list, dict]:
    """Renglones (cajas del detector) -> (cajas en orden de lectura, clase de zona de cada una o None, resumen)."""
    regs = regiones(img)
    if not regs or not lineas:
        return lineas, [None] * len(lineas), {}

    def solape(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    # cada renglón a la región que más lo cubre (la mitad de su área como poco)
    suya = []
    for c in lineas:
        area = max(1.0, (c[2] - c[0]) * (c[3] - c[1]))
        mejor = max(range(len(regs)), key=lambda r: solape(c, regs[r]["caja"]))
        suya.append(mejor if solape(c, regs[mejor]["caja"]) >= 0.5 * area else None)
    # regiones del corte: las de texto con renglones; los renglones sin región van como regiones de un renglón
    bloques, clases_bloque = [], []
    for r, g in enumerate(regs):
        miembros = [i for i, s in enumerate(suya) if s == r]
        if miembros:
            bloques.append((regs[r]["caja"], miembros, g["clase"]))
    for i, s in enumerate(suya):
        if s is None:
            bloques.append((lineas[i], [i], None))
    texto = [k for k, b in enumerate(bloques) if b[2] not in FUERA_DEL_CORTE]
    cajas_b = [b[0] for b in bloques]
    orden_b = _corte_xy(cajas_b, texto)
    arriba = [k for k, b in enumerate(bloques) if b[2] in ("PageHeader",)]
    abajo = [k for k, b in enumerate(bloques) if b[2] in FUERA_DEL_CORTE and k not in arriba]
    orden_cajas, clases = [], []
    for k in arriba + orden_b + abajo:
        caja, miembros, cl = bloques[k]
        for i in sorted(miembros, key=lambda i: (lineas[i][1], lineas[i][0])):
            orden_cajas.append(lineas[i])
            clases.append(CLASE.get(cl) if cl else None)
    resumen = {"regiones": len(regs), "figuras": sum(r["clase"] == "Figure" for r in regs),
               "renglones_en_figuras": sum(1 for i, s in enumerate(suya) if s is not None and regs[s]["clase"] == "Figure")}
    return orden_cajas, clases, resumen
