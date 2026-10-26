"""Bancos del orden de lectura: los seis documentos de las comparativas del 25-10-2026, siempre con Amanuensis Ligero.

    python3 bench/orden_bancos.py [--etiqueta base] [--env AMANUENSIS_ORDEN=gcn ...] [--solo grot,grice] [--paginas]

Corre el código de `src/` (no la herramienta instalada) con el intérprete de `amanu` y mide con rapidfuzz:
- Grotowski (revista a tres columnas, capa de texto ajena: --modo ocr) y Grice: WER por página frente a Gemini.
- Implicaturas (Pragmática para hispanistas, capa ajena: --modo ocr): WER frente a la referencia limpia arbitrada.
- Pragmática (escandell2) y La comunicación (escandell1): WER del documento frente a Gemini Pro.
- Anchieta (muestra.tif, portugués, español y tupí): WER frente a la edición.
Los datos viven en datos/bancos_orden (fuera del repositorio). Las salidas, en /tmp/bancos_orden/<etiqueta>/.
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import pathlib
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from rapidfuzz.distance import Levenshtein as L

AQUI = pathlib.Path(__file__).resolve().parent
SRC = AQUI.parent / "src"
DATOS = pathlib.Path(os.environ.get("BANCOS_ORDEN", pathlib.Path.home() / "Developer/eg2-ocr/datos/bancos_orden"))

# nombre -> (entrada, opciones, cifra de referencia del 25-10-2026 en WER %)
BANCOS = {
    "grot": ("grot/grot.pdf", ["--modo", "ocr", "--fiel"], 12.61),
    "implicaturas": ("implicaturas/doc.pdf", ["--modo", "ocr"], 0.64),
    "pragmatica": ("pragmatica/escandell2.pdf", [], 0.19),
    "comunicacion": ("comunicacion/escandell1.pdf", [], 2.31),
    "anchieta": ("anchieta/muestra.tif", [], 4.84),
    "grice": ("grice/grice.pdf", ["--fiel"], 0.56),
}


def interprete() -> str:
    amanu = shutil.which("amanu")
    if amanu:
        primera = open(amanu).readline()
        if primera.startswith("#!"):
            return primera[2:].strip()
    return sys.executable


# ---------------------------------------------------------------- normalizaciones (las de cada banco)
def norm_gemini(t):  # Grotowski y Grice
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S); t = re.sub(r"^[ \t]*#+[ \t]*", "", t, flags=re.M)
    t = re.sub(r"^[ \t]*[-*][ \t]+", "", t, flags=re.M)
    t = re.sub(r"\[\^\d+\]:?", " ", t); t = re.sub(r"\^\d+", " ", t); t = t.replace("**", "").replace("*", "").replace("_", "")
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"'); t = re.sub(r"(\w)-\s+(\w)", r"\1\2", t)
    return re.sub(r"\s+", " ", t).strip()


SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def norm_limpia(t):  # Implicaturas
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S); t = re.sub(r"^[ \t]*#+[ \t]*", "", t, flags=re.M)
    t = re.sub(r"^[ \t]*-[ \t]+", "", t, flags=re.M); t = t.replace("**", "").replace("*", "").replace("_", "")
    t = t.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", lambda m: "^" + m.group(0).translate(SUP), t)
    return re.sub(r"\s+", " ", t).strip()


def norm_ref(t):  # Pragmática, La comunicación y Anchieta (medir.py)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S); t = re.sub(r"^\s*#+\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*[-*]\s+", "", t, flags=re.M); t = re.sub(r"\[\^\d+\]:?", " ", t)
    t = t.replace("**", "").replace("*", "").replace("_", ""); t = re.sub(r"(\w)[-¬]\s*\n\s*(\w)", r"\1\2", t)
    t = t.replace("\xad", "")
    return re.sub(r"\s+", " ", t).strip()


def palabras(t):
    return re.findall(r"\w+", t.lower())


def wer(ref: list, hip: list) -> tuple[int, int]:
    return L.distance(ref, hip), len(ref)


def por_paginas(fiel: str, gem: str, pagina_de) -> tuple[dict, tuple[int, int]]:
    """WER por página (comentarios «página N» del Markdown fiel) frente a los .md de Gemini."""
    trozos = re.split(r"<!-- página (\d+) -->", fiel)
    A = {int(trozos[i]): trozos[i + 1] for i in range(1, len(trozos), 2)}
    G = {pagina_de(f): open(f).read() for f in glob.glob(f"{gem}/*.md")}
    comunes = sorted(set(A) & set(G))
    filas, d, n = {}, 0, 0
    for k in comunes:
        wg, wa = norm_gemini(G[k]).lower().split(), norm_gemini(A[k]).lower().split()
        e, m = wer(wg, wa)
        filas[k] = 100 * e / max(1, m)
        d += e; n += m
    return filas, (d, n)


def medir(nombre: str, md: pathlib.Path):
    t = md.read_text()
    if nombre == "grot":
        return por_paginas(t, DATOS / "grot/gem", lambda f: int(f.split("-")[-1][:3]) + 1)
    if nombre == "grice":
        # la cifra del banco es la del texto entero de las páginas que Gemini sí leyó (no la media de páginas)
        filas, _ = por_paginas(t, DATOS / "grice/gem", lambda f: int(f.split("-")[-1][:2]))
        trozos = re.split(r"<!-- página (\d+) -->", t)
        A = {int(trozos[i]): trozos[i + 1] for i in range(1, len(trozos), 2)}
        G = {int(f.split("-")[-1][:2]): open(f).read() for f in glob.glob(f"{DATOS}/grice/gem/*.md")}
        c = sorted(set(A) & set(G))
        tg = " ".join(norm_gemini(G[k]) for k in c).split(); ta = " ".join(norm_gemini(A[k]) for k in c).split()
        return filas, wer(tg, ta)
    if nombre == "implicaturas":
        r = norm_limpia((DATOS / "implicaturas/ref_limpia.md").read_text()).split()
        return {}, wer(r, norm_limpia(t).split())
    if nombre in ("pragmatica", "comunicacion"):
        refs = sorted((DATOS / nombre / "ref").glob("*.md"))
        a = norm_ref("\n\n".join(r.read_text().replace("<<<SIGUE>>>", "") for r in refs))
        return {}, wer(palabras(a), palabras(norm_ref(t)))
    if nombre == "anchieta":
        ref = " ".join(norm_ref(r.read_text()) for r in sorted((DATOS / "anchieta/ref").glob("*.txt")))
        return {}, wer(palabras(ref), palabras(norm_ref(t)))
    raise ValueError(nombre)


def correr(nombre: str, salida: pathlib.Path, env: dict, py: str) -> pathlib.Path:
    entrada, opciones, _ = BANCOS[nombre]
    ruta = DATOS / entrada
    out = salida / nombre
    out.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, PYTHONPATH=str(SRC), **env)
    r = subprocess.run([py, "-m", "amanuensis", str(ruta), "-m", "ligero", *opciones, "-o", str(out), "-q"],
                       env=e, capture_output=True, text=True)
    md = out / (ruta.stem + ".md")
    if r.returncode != 0 or not md.exists():
        raise RuntimeError(f"{nombre}: {r.stderr[-2000:]}")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiqueta", default="base")
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--solo", default="")
    ap.add_argument("--paginas", action="store_true", help="WER de cada página (Grotowski y Grice)")
    ap.add_argument("--hilos", type=int, default=3)
    a = ap.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    nombres = [n for n in BANCOS if not a.solo or n in a.solo.split(",")]
    salida = pathlib.Path("/tmp/bancos_orden") / a.etiqueta
    py = interprete()
    with ThreadPoolExecutor(a.hilos) as ex:
        mds = dict(zip(nombres, ex.map(lambda n: correr(n, salida, env, py), nombres)))
    print(f"\n{a.etiqueta} {' '.join(a.env)}")
    print(f"{'banco':14} {'WER %':>7} {'(ref.)':>7}  palabras")
    for n in nombres:
        filas, (d, m) = medir(n, mds[n])
        print(f"{n:14} {100 * d / max(1, m):7.2f} {BANCOS[n][2]:7.2f}  {d}/{m}")
        if a.paginas and filas:
            print("    " + "  ".join(f"p{k} {v:.1f}" for k, v in filas.items()))


if __name__ == "__main__":
    main()
