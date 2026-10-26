"""La doble página: un libro escaneado abierto (dos hojas lado a lado en una imagen apaisada) se parte por el canal
antes del OCR. Leída entera, el orden de lectura mezcla los renglones de las dos hojas; partida, cada hoja se lee como
lo que es.

La medida es el número de cambios entre tinta y papel por columna: alto donde hay texto, casi nulo en el canal (sea
blanco o la sombra oscura del lomo, que tampoco alterna). Calibrado con un libro escaneado de 18 dobles páginas: el
canal tiene entre el 1 % y el 6 % de los cambios de las hojas; una página de texto corrido que cruza el centro no baja
nunca de ahí. Una página sola escaneada de lado también es apaisada, y un blanco entre párrafos que caiga en el centro
parece un canal: si el perfil de la tinta dice que los renglones van en vertical, no se parte. Es el mismo algoritmo que crates/yappy-core/src/ocr/doble.rs (Yappy) y web/prueba/src/doble.ts (demo).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

APAISADO = 1.2  # ancho / alto a partir del cual una imagen puede ser una doble página
CANAL = (0.38, 0.62)  # franja central donde se busca el canal (fracción del ancho)
VALLE = 0.15  # el canal tiene que bajar de esta fracción de los cambios de la hoja más pobre
TINTA_MIN = 0.01  # cambios por columna mínimos de cada hoja, en fracción del alto (si no, hoja en blanco)
MUESTRA = 1200  # columnas muestreadas como mucho
RAZON_VERTICAL = 0.75  # nitidez filas / columnas por debajo de la cual los renglones van en vertical (en las dos hojas)


def _otsu(m: np.ndarray) -> int:
    hist = np.bincount(m.ravel(), minlength=256).astype(np.float64)
    total, suma = hist.sum(), (np.arange(256) * hist).sum()
    peso_b = suma_b = mejor = 0.0
    umbral = 128
    for t in range(256):
        peso_b += hist[t]
        if peso_b == 0 or peso_b == total:
            continue
        suma_b += t * hist[t]
        mb, mf = suma_b / peso_b, (suma - suma_b) / (total - peso_b)
        v = peso_b * (total - peso_b) * (mb - mf) ** 2
        if v > mejor:
            mejor, umbral = v, t
    return umbral


def _media_movil(v: np.ndarray, k: int) -> np.ndarray:
    m = k // 2
    c = np.concatenate([[0.0], np.cumsum(v)])
    i = np.arange(len(v))
    a, b = np.maximum(i - m, 0), np.minimum(i + m + 1, len(v))
    return (c[b] - c[a]) / (b - a)


def _mediana(v: np.ndarray) -> float:
    return float(np.sort(v)[len(v) // 2]) if len(v) else 0.0


def _tinta_reducida(img: Image.Image, lado: int = 1000) -> np.ndarray:
    """Tinta binaria normalizada por el fondo (máximo 9×9 y caja de 41×41), umbral 0,22, a 1000 px de lado."""
    g = img.convert("L")
    s = min(1.0, lado / max(g.size))
    if s < 1:
        g = g.resize((max(1, int(g.size[0] * s)), max(1, int(g.size[1] * s))), Image.BILINEAR)
    fondo = np.asarray(g.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.BoxBlur(20)), np.float32)
    return (1 - np.asarray(g, np.float32) / np.maximum(fondo, 1)) > 0.22


def _razon(b: np.ndarray) -> float:
    """Nitidez del perfil de filas frente a la del de columnas (media del cuadrado de la diferencia entre vecinos, relativa
    a la media del perfil al cuadrado): alta con renglones horizontales, baja con renglones en vertical."""
    b = b.astype(np.float64)
    ph, pv = b.sum(1), b.sum(0)
    if len(ph) < 2 or len(pv) < 2:
        return float("inf")
    h = np.mean(np.diff(ph) ** 2) / max(1e-9, np.mean(ph) ** 2)
    v = np.mean(np.diff(pv) ** 2) / max(1e-9, np.mean(pv) ** 2)
    return float(h / max(v, 1e-12))


def parece_vertical(img: Image.Image, x: int | None = None) -> bool:
    """¿Renglones en vertical (una página sola escaneada de lado)? Se mide en cada hoja por separado (las dos mitades
    por `x`): en un libro abierto combado los renglones de las dos hojas no casan y el perfil de la página entera se
    emborrona. Medido en 43 imágenes con canal (32 dobles páginas de libros escaneados y páginas sueltas
    giradas 90°): las giradas no pasan de 0,48 en su mejor hoja; las dobles no bajan de 1,14."""
    b = _tinta_reducida(img)
    xs = b.shape[1] // 2 if x is None else int(round(x * b.shape[1] / img.size[0]))
    mitades = [m for m in (b[:, :xs], b[:, xs:]) if m.shape[1] >= 8]
    return bool(mitades) and max(_razon(m) for m in mitades) < RAZON_VERTICAL


def canal(img: Image.Image) -> int | None:
    """Si `img` es una doble página, la columna (px) por la que partirla; si no, None."""
    ancho, alto = img.size
    if alto == 0 or ancho < alto * APAISADO:
        return None
    paso = max(-(-ancho // MUESTRA), 1)
    g = np.asarray(img.convert("L"))
    m = g[::paso, ::paso]  # filas y columnas muestreadas, como el original en Rust
    umbral = _otsu(m)
    tinta = m <= umbral
    cambios = (tinta[1:] != tinta[:-1]).sum(axis=0).astype(np.float64)
    n = len(cambios)
    suave = _media_movil(cambios, max(int(n * 0.02), 3))

    def franja(a, b):
        return int(n * a), int(n * b)

    lo, hi = franja(*CANAL)
    if hi <= lo:
        return None
    i_canal = lo + int(np.argmin(suave[lo:hi]))
    valle = suave[i_canal]
    hoja = min(_mediana(suave[slice(*franja(0.10, 0.40))]), _mediana(suave[slice(*franja(0.60, 0.90))]))
    filas = -(-alto // paso)
    if hoja < filas * TINTA_MIN or valle > hoja * VALLE:
        return None
    if parece_vertical(img, i_canal * paso):  # una página sola girada 90°: apaisada y con un blanco en el centro, pero no es un libro
        return None
    return i_canal * paso


def partir(img: Image.Image) -> tuple[int, Image.Image, Image.Image] | None:
    """(columna, izquierda, derecha) si `img` es una doble página; None si no."""
    x = canal(img)
    if x is None:
        return None
    w, h = img.size
    return x, img.crop((0, 0, x, h)), img.crop((x, 0, w, h))
