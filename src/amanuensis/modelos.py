"""Catálogo y descarga automática de pesos desde el almacén público (Cloudflare R2).

Caché: ~/.cache/amanuensis/<carpeta>/ (o AMANUENSIS_CACHE). Origen: AMANUENSIS_ORIGEN sustituye a la base del
catálogo. Una carpeta local con los mismos ficheros: AMANUENSIS_MODELOS=/ruta (subcarpetas por modelo).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from importlib import resources
from pathlib import Path


def entorno(nombre: str, defecto=None):
    """AMANUENSIS_<nombre> (o EG2OCR_<nombre>, el nombre antiguo)."""
    return os.environ.get("AMANUENSIS_" + nombre) or os.environ.get("EG2OCR_" + nombre) or defecto


def catalogo() -> dict:
    return json.loads(resources.files("amanuensis").joinpath("catalogo.json").read_text())


def cache() -> Path:
    return Path(entorno("CACHE", Path.home() / ".cache" / "amanuensis"))


def _local(m: dict) -> Path | None:
    """Pesos aún no publicados: carpeta «local» del catálogo dentro del repositorio eg2-ocr."""
    if not m.get("local"):
        return None
    for r in (entorno("REPO"),):
        if r and all((Path(r) / m["local"] / f).exists() for f in m["ficheros"]):
            return Path(r) / m["local"]
    return None


def disponible(nombre: str) -> bool:
    m = catalogo()["modelos"][nombre]
    if _local(m):
        return True
    if entorno("MODELOS"):
        p = Path(entorno("MODELOS")) / m["carpeta"]
        if all((p / f).exists() for f in m["ficheros"]):
            return True
    if all((cache() / m["carpeta"] / f).exists() for f in m["ficheros"]):
        return True
    return not m.get("pendiente")


def _bajar(url: str, destino: Path, tam: int):
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    destino.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "amanuensis"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or tam or 0)
        hecho, ultimo = 0, -1
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            hecho += len(b)
            if total and sys.stderr.isatty():
                pct = int(100 * hecho / total)
                if pct != ultimo:
                    print(f"\r  {destino.name}: {pct} % de {total / 1e6:.0f} MB", end="", file=sys.stderr)
                    ultimo = pct
    if sys.stderr.isatty():
        print(file=sys.stderr)
    real = tmp.stat().st_size
    if tam and real != tam:
        tmp.unlink()
        raise OSError(f"{url}: tamaño inesperado ({real} en vez de {tam})")
    os.replace(tmp, destino)


def ruta(nombre: str) -> str:
    """Carpeta local del modelo; la descarga la primera vez."""
    cat = catalogo()
    m = cat["modelos"][nombre]
    if entorno("MODELOS"):
        p = Path(entorno("MODELOS")) / m["carpeta"]
        if all((p / f).exists() for f in m["ficheros"]):
            return str(p)
    if _local(m) and not all((cache() / m["carpeta"] / f).exists() for f in m["ficheros"]):
        return str(_local(m))
    destino = cache() / m["carpeta"]
    faltan = [f for f in m["ficheros"] if not (destino / f).exists()]
    if faltan and m.get("pendiente"):
        raise FileNotFoundError(f"el modelo «{nombre}» aún no está publicado")
    base = entorno("ORIGEN", cat["base"]).rstrip("/")
    for f in faltan:
        print(f"descargando {m['carpeta']}/{f} (solo la primera vez)", file=sys.stderr)
        spec = m["ficheros"][f]
        if isinstance(spec, dict):  # fichero servido en partes (el almacén limita el tamaño por objeto)
            partes = []
            for k in range(spec["partes"]):
                p = destino / f"{f}.parte{k}"
                if not p.exists():
                    _bajar(f"{base}/{m['carpeta']}/{f}.parte{k}", p, 0)
                partes.append(p)
            tmp = destino / (f + ".parcial")
            with open(tmp, "wb") as o:
                for p in partes:
                    o.write(p.read_bytes())
            if tmp.stat().st_size != spec["bytes"]:
                tmp.unlink()
                raise OSError(f"{f}: tamaño inesperado tras unir las partes")
            os.replace(tmp, destino / f)
            for p in partes:
                p.unlink()
        else:
            _bajar(f"{base}/{m['carpeta']}/{f}", destino / f, spec)
    return str(destino)
