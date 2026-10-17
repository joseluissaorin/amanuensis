"""PDF: páginas como imagen (pypdfium2) y capa de texto con cajas de línea."""
from __future__ import annotations

import unicodedata

import numpy as np
from PIL import Image


def abrir(ruta: str, contrasena: str | None = None):
    import pypdfium2 as pdfium

    return pdfium.PdfDocument(ruta, password=contrasena)


def imagen(pag, dpi: float, color: bool = False) -> Image.Image:
    im = pag.render(scale=dpi / 72.0, grayscale=not color).to_pil()
    return im.convert("RGB" if color else "L")


def lineas_texto(pag, dpi: float):
    """Líneas de la capa de texto: [(caja en px a `dpi`, texto)], en el orden del PDF. Se recorre
    carácter a carácter (pdfium marca los saltos de línea con \\r\\n) y se cortan además las líneas en los
    saltos horizontales grandes (columnas que el PDF escribe en la misma línea)."""
    tp = pag.get_textpage()
    _, alto_pt = pag.get_size()
    s = dpi / 72.0
    n = tp.count_chars()
    texto = (tp.get_text_range(0, n) if n else "").replace("\ufffe", "-")  # pdfium marca así el guion de corte
    lineas, act, caja = [], [], None

    def cerrar():
        nonlocal act, caja
        t = " ".join("".join(act).split())
        if t and caja:
            lineas.append(([caja[0] * s, (alto_pt - caja[3]) * s, caja[2] * s, (alto_pt - caja[1]) * s], t))
        act, caja = [], None

    for i, ch in enumerate(texto[:n]):
        if ch in "\r\n":
            cerrar()
            continue
        if ch.isspace():
            act.append(" ")
            continue
        l, b, r, t = tp.get_charbox(i)
        if r - l <= 0 or t - b <= 0:
            act.append(ch)
            continue
        if caja is not None:
            h = max(1e-3, caja[3] - caja[1])
            otra_fila = min(t, caja[3]) - max(b, caja[1]) < 0.3 * min(h, t - b)
            if otra_fila or l - caja[2] > 2.5 * h or l < caja[0] - 0.5 * h:
                cerrar()
        if caja is not None and l - caja[2] > 0.25 * (caja[3] - caja[1]) and act and act[-1] != " ":
            act.append(" ")
        act.append(ch)
        caja = [l, b, r, t] if caja is None else [min(caja[0], l), min(caja[1], b), max(caja[2], r), max(caja[3], t)]
    cerrar()
    tp.close()
    return lineas


def capa_fiable(lineas) -> tuple[bool, str]:
    """Comprobaciones baratas de la capa de texto (sin OCR)."""
    texto = "".join(t for _, t in lineas)
    if len(texto) < 20:
        return False, "sin capa de texto"
    malos = sum(1 for c in texto if c == "�" or unicodedata.category(c) in ("Co", "Cc", "Cn"))
    if malos > 0.02 * len(texto):
        return False, f"capa de texto con {malos} caracteres ilegibles"
    letras = sum(c.isalpha() for c in texto)
    if letras < 0.4 * len(texto.replace(" ", "")):
        return False, "capa de texto con pocas letras (¿fuentes sin tabla Unicode?)"
    return True, "capa de texto"


def cer(a: str, b: str) -> float:
    import difflib

    if not b:
        return 1.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return 1.0 - sm.ratio()
