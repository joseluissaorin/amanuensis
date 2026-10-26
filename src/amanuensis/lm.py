"""Modelos de lengua de palabras para la voz (KenLM de 4-gramas por lengua) y búsqueda en haz CTC.

Dependencia opcional: `pip install "amanuensis[voz-lm]"` (pyctcdecode y kenlm). Con Python 3.13 o posterior, el
kenlm de PyPI no compila (su C++ pregenerado es antiguo): `python -m amanuensis.lm instalar` lo compila desde la
fuente regenerando la parte de Cython (hace falta un compilador de C++ y CMake).

Pesos: modelo «voz-lm» del catálogo. `ajuste.json` (alfa y beta por lengua, ajustados en validación) se baja con el
modelo; cada lengua (`<l>.bin` y `<l>.unigramas.json`) se baja la primera vez que se usa. Carpeta local con los
mismos ficheros: AMANUENSIS_VOZ_LM_DIR.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from .modelos import _bajar, cache, catalogo, entorno

# Lenguas cuyo LM se baja solo con el modo «auto» (las del proyecto); el resto, con lm="si"
PRINCIPALES = ("es", "ca", "gl", "eu", "pt")


def disponible() -> bool:
    try:
        import kenlm  # noqa: F401
        import pyctcdecode  # noqa: F401
        return True
    except Exception:
        return False


def _entrada():
    return catalogo()["modelos"].get("voz-lm")


def carpeta() -> Path | None:
    if entorno("VOZ_LM_DIR"):
        return Path(entorno("VOZ_LM_DIR"))
    m = _entrada()
    if not m:
        return None
    from .modelos import _local

    loc = _local(m)  # aún sin publicar: la carpeta del repositorio (AMANUENSIS_REPO)
    if loc is not None and not (cache() / m["carpeta"] / "ajuste.json").exists():
        return loc
    return cache() / m["carpeta"]


def ajuste() -> dict:
    c = carpeta()
    if c is None:
        return {}
    f = c / "ajuste.json"
    if not f.exists():
        m = _entrada()
        if not m:
            return {}
        base = entorno("ORIGEN", catalogo()["base"]).rstrip("/")
        _bajar(f"{base}/{m['carpeta']}/ajuste.json", f, m["ficheros"]["ajuste.json"])
    return json.loads(f.read_text())


def lenguas() -> list[str]:
    m = _entrada()
    if m:
        return sorted(m.get("lenguas", {}))
    c = carpeta()
    return sorted(p.stem for p in c.glob("*.bin")) if c and c.exists() else []


def ficheros_lengua(l: str) -> tuple[Path, Path] | None:
    """(<l>.bin, <l>.unigramas.json), bajándolos si hace falta; None si no hay LM para esa lengua."""
    c = carpeta()
    if c is None:
        return None
    b, u = c / f"{l}.bin", c / f"{l}.unigramas.json"
    if b.exists() and u.exists():
        return b, u
    m = _entrada()
    if not m or l not in m.get("lenguas", {}):
        return None
    base = entorno("ORIGEN", catalogo()["base"]).rstrip("/") + "/" + m["carpeta"]
    for nombre, spec in m["lenguas"][l].items():
        destino = c / nombre
        if destino.exists():
            continue
        print(f"descargando el modelo de lengua {nombre} (solo la primera vez)", file=sys.stderr)
        if isinstance(spec, dict):
            partes = []
            for k in range(spec["partes"]):
                p = c / f"{nombre}.parte{k}"
                if not p.exists():
                    _bajar(f"{base}/{nombre}.parte{k}", p, 0)
                partes.append(p)
            tmp = c / (nombre + ".parcial")
            with open(tmp, "wb") as o:
                for p in partes:
                    with open(p, "rb") as fp:
                        while blk := fp.read(1 << 24):
                            o.write(blk)
            if tmp.stat().st_size != spec["bytes"]:
                tmp.unlink()
                raise OSError(f"{nombre}: tamaño inesperado tras unir las partes")
            os.replace(tmp, destino)
            for p in partes:
                p.unlink()
        else:
            _bajar(f"{base}/{nombre}", destino, spec)
    return b, u


_DEC: dict = {}


def decodificador(tokens: list[str], l: str):
    """pyctcdecode con el KenLM de la lengua y su alfa/beta; None si no hay LM."""
    if l in _DEC:
        return _DEC[l]
    _DEC.clear()  # un LM a la vez: cada uno ocupa cientos de MB
    fs = ficheros_lengua(l)
    if fs is None:
        _DEC[l] = None
        return None
    import logging

    logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
    from pyctcdecode import build_ctcdecoder

    aj = ajuste().get(l, {"alfa": 0.5, "beta": 1.0})
    etiquetas = [""] + [t if t != "<unk>" else "▁⁇▁" for t in tokens[1:]]
    uni = json.loads(fs[1].read_text())
    _DEC[l] = build_ctcdecoder(etiquetas, kenlm_model_path=str(fs[0]), unigrams=uni, alpha=aj["alfa"], beta=aj["beta"])
    return _DEC[l]


def haz(dec, logp, ancho: int = 16):
    """[(palabra, marco_inicio, marco_fin)] de la mejor hipótesis (logp: [T, V] log-probabilidades).
    prune_history=True, como pyctcdecode.decode() (con él se ajustaron alfa y beta y se midió: sin él el haz pierde
    palabras cortas como «a» o «e»). Palabras del texto del haz y marcos de text_frames; si alguna no tuviera marcos,
    se le da el hueco entre sus vecinas."""
    import numpy as np

    b = dec.decode_beams(np.ascontiguousarray(logp, dtype=np.float32), beam_width=ancho, prune_history=True)[0]  # como decode(): es lo que se ajustó y midió
    pal = b[0].split()
    fr = [(w, int(s), int(e)) for w, (s, e) in b[2]]
    out = [None] * len(pal)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, pal, [w for w, _, _ in fr], autojunk=False).get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for i, j in zip(range(i1, i2), range(j1, j2)):
                out[i] = (pal[i], fr[j][1], fr[j][2])
    for i, x in enumerate(out):
        if x is None:
            a = next((out[k][2] for k in range(i - 1, -1, -1) if out[k]), 0)
            z = next((out[k][1] for k in range(i + 1, len(out)) if out[k]), max(a + 1, len(logp)))
            out[i] = (pal[i], a, max(a + 1, min(z, a + 2)))
    return out


# ---------- mayúsculas y puntuación de la salida «rica» sobre las palabras del LM ----------
_APOS = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'"})
_BORRA = re.compile(r"[^\w' ]+", re.UNICODE)


def plano(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).translate(_APOS).lower().replace("_", " ")
    return re.sub(r"\s+", " ", _BORRA.sub(" ", t)).strip(" '")


def _forma(rico: str, palabra: str) -> str:
    """Lleva a `palabra` (minúsculas, sin puntuación) la puntuación de los bordes y las mayúsculas de `rico`."""
    m = re.match(r"^(\W*)(.*?)(\W*)$", rico, re.UNICODE)
    pre, nucleo, post = m.group(1), m.group(2), m.group(3)
    if nucleo and nucleo.isupper() and len(nucleo) > 1:
        palabra = palabra.upper()
    elif nucleo[:1].isupper():
        palabra = palabra[:1].upper() + palabra[1:]
    return pre + palabra + post


def fusionar(lm: list[dict], rico: list[dict]) -> list[dict]:
    """Palabras del LM (mejor léxico) con la forma de la salida rica (mayúsculas, puntuación) donde se alinean.
    Cada palabra: {texto, inicio, fin, conf}. Las del LM conservan sus tiempos."""
    a = [plano(w["texto"]) for w in lm]
    b = [plano(w["texto"]) for w in rico]
    out = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                out.append({**lm[i], "texto": rico[j]["texto"], "conf": rico[j].get("conf", lm[i].get("conf"))})
        elif op == "replace" and i2 - i1 == j2 - j1:
            for i, j in zip(range(i1, i2), range(j1, j2)):
                out.append({**lm[i], "texto": _forma(rico[j]["texto"], lm[i]["texto"])})
        elif op in ("replace", "delete"):
            for i in range(i1, i2):
                out.append(dict(lm[i]))
            if op == "replace" and j2 > j1:  # la puntuación final del tramo rico pasa a la última palabra
                post = re.search(r"[.,;:?!…]+$", rico[j2 - 1]["texto"])
                if post and out:
                    out[-1]["texto"] += post.group(0)
    # mayúscula tras punto (y al principio) en las palabras que no vienen de la salida rica
    fin = True
    for w in out:
        t = w["texto"]
        if fin and t[:1].islower():
            w["texto"] = t[:1].upper() + t[1:]
        fin = t[-1:] in ".?!…"
    return out


def instalar():
    """Compila kenlm desde la fuente (Cython regenerado) para el intérprete actual."""
    import shutil
    import subprocess
    import tempfile
    import urllib.request
    import zipfile

    with tempfile.TemporaryDirectory() as d:
        z = os.path.join(d, "kenlm.zip")
        urllib.request.urlretrieve("https://github.com/kpu/kenlm/archive/master.zip", z)
        zipfile.ZipFile(z).extractall(d)
        src = os.path.join(d, "kenlm-master")
        uv = shutil.which("uv")
        pip = [uv, "pip", "install", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install"]
        subprocess.run(pip + ["cython", "pyctcdecode"], check=True)
        subprocess.run([sys.executable, "-m", "cython", "--cplus", "python/kenlm.pyx", "-o", "python/kenlm.cpp"], cwd=src, check=True)
        subprocess.run(pip + [src], check=True)
    print("kenlm instalado:", disponible())


if __name__ == "__main__":
    if sys.argv[1:2] == ["instalar"]:
        instalar()
    else:
        print("uso: python -m amanuensis.lm instalar")
