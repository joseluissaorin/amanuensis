"""Página derecha y enderezada antes de detectar líneas (vía de imagen; los PDF digitales no lo necesitan).

- Inclinación: el ángulo (±5°) que hace más nítido el perfil horizontal de tinta en una miniatura; búsqueda a
  pasos de 0,5° y luego de 0,1°; solo se corrige a partir de 0,3°.
- Giro de 90, 180 o 270°: si las líneas detectadas salen más altas que anchas, la página está tumbada; se prueban
  los dos cuartos de vuelta. Entre una página derecha y una cabeza abajo decide el propio lector: se leen las líneas
  más largas en los dos sentidos y gana la lectura con más confianza (solo si la diferencia es clara).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def inclinacion(img: Image.Image, lado: int = 600, maximo: float = 5.0) -> float:
    g = img.convert("L")
    s = lado / max(g.size)
    if s < 1:
        g = g.resize((max(1, int(g.size[0] * s)), max(1, int(g.size[1] * s))), Image.BILINEAR)
    tinta = Image.fromarray((255 - np.asarray(g)).astype(np.uint8))
    if np.asarray(tinta).mean() < 1.0:
        return 0.0

    def nitidez(a):
        p = np.asarray(tinta.rotate(a, resample=Image.BILINEAR), np.float32).sum(1)
        return float(((p[1:] - p[:-1]) ** 2).sum())

    mejor = max(np.arange(-maximo, maximo + 1e-6, 1.0), key=nitidez)
    mejor = max(np.arange(mejor - 0.75, mejor + 0.75 + 1e-6, 0.25), key=nitidez)
    return float(max(np.arange(mejor - 0.2, mejor + 0.2 + 1e-6, 0.1), key=nitidez))


def enderezar(img: Image.Image, minimo: float = 0.3):
    a = inclinacion(img)
    if abs(a) < minimo:
        return img, 0.0
    fondo = 255 if img.mode == "L" else (255, 255, 255)
    return img.rotate(a, resample=Image.BICUBIC, expand=False, fillcolor=fondo), round(a, 2)


def _girar_cajas_180(cs, W, H):
    return [[W - c[2], H - c[3], W - c[0], H - c[1]] for c in cs]


VACIAS = set("""de la que el en y a los se del las un por con no una su para es al lo como más o pero sus le ha me si sin sobre
este ya entre cuando todo esta ser son dos también fue había era muy años hasta desde está mi porque qué sólo han yo
the of and to in is that for it as was with be by on not he this are or his from at which but have an they
et in est non ad cum quod qui ut sed per de le les des du et la un une est pour dans que qui il ne pas
der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er
e o da do em um para com não uma os no se na por mais as dos como mas foi ao ele das à seu sua ou ser""".split())


def verosimil(textos, confs) -> float:
    """Cuán probable es que unas lecturas sean texto derecho: confianza media + proporción de palabras vacías
    frecuentes (es, en, la, fr, de, pt) + proporción de letras. Del revés, el lector lee con seguridad, pero basura."""
    pal = [w.strip(".,;:¡!¿?()«»\"'").lower() for t in textos for w in t.split()]
    vac = sum(w in VACIAS for w in pal) / max(1, len(pal))
    letras = sum(c.isalpha() for t in textos for c in t) / max(1, sum(len(t.replace(" ", "")) for t in textos))
    return float(np.mean(confs)) + 2.0 * vac + letras if confs else 0.0


def _puntuar(im, cs, leer, n=8):
    from .imagen import recorte_de_caja

    largas = sorted(cs, key=lambda c: -(c[2] - c[0]))[:n]
    if not largas:
        return 0.0
    ls = leer([recorte_de_caja(im, c) for c in largas])
    return verosimil([l.texto for l in ls], [l.conf for l in ls])


def _girar_cajas_180(cs, W, H):
    return [[W - c[2], H - c[3], W - c[0], H - c[1]] for c in cs]


def tumbada(cajas) -> bool:
    return len(cajas) < 8 or float(np.median([(c[2] - c[0]) / max(1.0, c[3] - c[1]) for c in cajas])) < 1.0


def orientar(img: Image.Image, cajas, detectar, leer):
    """Devuelve (imagen derecha, grados de giro antihorario, cajas). Página derecha o cabeza abajo: se comparan las
    líneas más largas leídas en los dos sentidos (las cajas giradas, sin volver a detectar). Página tumbada: se
    prueban 90 y 270 (y su media vuelta)."""
    candidatos = [(0, img, cajas)]
    # renglones diminutos (< 14 px): el lector no distingue derecho de revés y una media vuelta en falso destroza la página
    if cajas and float(np.median([c[3] - c[1] for c in cajas])) < 14 and not tumbada(cajas):
        return img, 0, cajas
    if tumbada(cajas):
        for g in (90, 270):
            im = img.rotate(g, expand=True)
            candidatos.append((g, im, detectar(im)))
    mejor = None
    for g, im, cs in candidatos:
        for extra in (0, 180):
            im2 = im if extra == 0 else im.rotate(180)
            cs2 = cs if extra == 0 else _girar_cajas_180(cs, *im.size)
            p = _puntuar(im2, cs2, leer) - (0.15 if extra else 0.0)  # cambiar de sentido tiene que ganar con claridad
            if mejor is None or p > mejor[0]:
                mejor = (p, (g + extra) % 360, im2, extra != 0)
    _, g, im, redetectar = mejor
    if g == 0:
        return img, 0, cajas
    return im, g, detectar(im)
