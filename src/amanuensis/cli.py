"""Amanuensis: lee cualquier página de la historia y escribe lo que le dictan (alias: amanu, eg2ocr).

  amanuensis documento.pdf                     # Markdown a la salida estándar
  amanuensis pagina.jpg -f tei -o salida/      # TEI en salida/pagina.tei.xml
  amanuensis carpeta/ -f page,alto,json -o x/  # varios formatos a la vez, todas las imágenes y PDF
  amanuensis escucha entrevista.mp3 -f srt     # voz a texto con tiempos (txt, srt, vtt, json)
  amanuensis modelos                           # catálogo y qué hay ya descargado
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import __version__
from .salidas import EXTENSION, FORMATOS

EXT_DOC = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".jp2", ".gif"}
EXT_AUDIO = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".mov", ".mkv", ".webm", ".aiff"}


def entradas(rutas, exts):
    for r in rutas:
        p = Path(r)
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.suffix.lower() in exts)
        elif p.exists():
            yield p
        else:
            print(f"no existe: {r}", file=sys.stderr)


def escribir(texto: str, salida: str | None, base: Path, ext: str, varios: bool):
    if not salida:
        sys.stdout.write(texto)
        if varios:
            sys.stdout.write("\n")
        return
    d = Path(salida)
    d.mkdir(parents=True, exist_ok=True)
    f = d / (base.stem + ext)
    f.write_text(texto, encoding="utf-8")
    print(f"  -> {f}", file=sys.stderr)


def cmd_escucha(argv):
    ap = argparse.ArgumentParser(prog="amanuensis escucha", description="voz a texto con tiempos por palabra")
    ap.add_argument("audios", nargs="+")
    ap.add_argument("-f", "--formato", default="txt", help="txt, srt, vtt, json (varios separados por comas)")
    ap.add_argument("-o", "--salida", help="carpeta de salida (por defecto, la salida estándar)")
    ap.add_argument("--hilos", type=int, default=0)
    a = ap.parse_args(argv)
    from . import modelos
    from .modelos import entorno
    from .escucha import Escucha, exportar

    carpeta = entorno("VOZ") or modelos.ruta("voz")
    e = Escucha(carpeta, a.hilos)
    fs = a.formato.split(",")
    lista = list(entradas(a.audios, EXT_AUDIO))
    for p in lista:
        t0 = time.time()
        r = e.transcribir(str(p))
        print(f"{p.name}: {r['duracion']} s de audio en {time.time() - t0:.1f} s (idioma {r['idioma']})", file=sys.stderr)
        for f in fs:
            escribir(exportar(r, f), a.salida, p, "." + f, len(lista) > 1 or len(fs) > 1)


def cmd_modelos():
    from . import modelos

    cat = modelos.catalogo()
    print(f"origen: {cat['base']}\ncaché: {modelos.cache()}")
    for n, m in cat["modelos"].items():
        d = modelos.cache() / m["carpeta"]
        hay = all((d / f).exists() for f in m["ficheros"])
        mb = sum(v["bytes"] if isinstance(v, dict) else v for v in m["ficheros"].values()) / 1e6
        estado = "descargado" if hay else "local (sin publicar)" if modelos._local(m) else ("pendiente de publicar" if m.get("pendiente") else f"se descarga al usarlo ({mb:.0f} MB)")
        nc = " [uso no comercial]" if "no comercial" in m.get("licencia", "") else ""
        print(f"  {n:<16} {estado:<34} {m['descripcion']}{nc}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "escucha":
        return cmd_escucha(argv[1:])
    if argv and argv[0] == "modelos":
        return cmd_modelos()
    if argv and argv[0] == "visor":
        from .visor import main as visor

        return visor(argv[1:])
    ap = argparse.ArgumentParser(prog="amanuensis", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entradas", nargs="+", help="PDF, imágenes o carpetas")
    ap.add_argument("-f", "--formato", default="md", help=f"{', '.join(FORMATOS)} (varios separados por comas)")
    ap.add_argument("-o", "--salida", help="carpeta de salida (por defecto, la salida estándar)")
    ap.add_argument("-m", "--modelo", default="auto", help="auto, ligero, minimo, historico, multilingue, impreso")
    ap.add_argument("-r", "--rapido", action="store_true", help="solo Amanuensis Ligero y el detector Mínimo (CPU, ~0,1-0,2 s por página)")
    ap.add_argument("--maqueta", default="auto", choices=["auto", "si", "no"], help="maquetación completa del proyecto (zonas, líneas con base, orden)")
    ap.add_argument("--modo", default="auto", choices=["auto", "ocr", "texto"], help="PDF: capa de texto si es fiable (auto), siempre OCR, o siempre la capa")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--imagenes", default="auto", choices=["auto", "si", "no"], help="extraer figuras, grabados, capitulares y sellos (con la maquetación completa)")
    ap.add_argument("--paginas", type=int, default=0, help="como mucho N páginas por fichero")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--hilos", type=int, default=0)
    ap.add_argument("-q", "--silencio", action="store_true")
    ap.add_argument("--version", action="version", version=f"amanuensis {__version__}")
    a = ap.parse_args(argv)
    a.modelo = {"mini": "minimo", "mini-base": "ligero", "mínimo": "minimo"}.get(a.modelo, a.modelo)
    fs = a.formato.split(",")
    for f in fs:
        if f not in FORMATOS:
            ap.error(f"formato desconocido: {f}")
    from .motor import Motor

    m = Motor(a.modelo, a.maqueta, a.modo, a.dpi, a.dispositivo, a.hilos, not a.silencio, a.rapido, a.imagenes)
    lista = list(entradas(a.entradas, EXT_DOC))
    fallos = 0
    for p in lista:
        try:
            t0 = time.time()
            d = m.leer(str(p), a.paginas)
            m.log(f"  {len(d.paginas)} páginas en {time.time() - t0:.1f} s")
            for f in fs:
                if f == "page" and a.salida and len(d.paginas) > 1:  # PAGE XML: un fichero por página
                    from .salidas import page

                    for i in range(len(d.paginas)):
                        escribir(page(d, i), a.salida, p, f"_p{i + 1:03d}.page.xml", True)
                    continue
                escribir(d.exportar(f), a.salida, p, EXTENSION[f], len(lista) > 1 or len(fs) > 1)
            if d.imagenes:
                if a.salida:
                    m.log(f"  -> {d.guardar_imagenes(a.salida)} imágenes en {Path(a.salida) / 'img' / p.stem}")
                else:
                    m.log("  aviso: hay imágenes extraídas; usa -o CARPETA para guardarlas")
        except KeyboardInterrupt:
            raise
        except Exception as e:  # sigue con los demás ficheros
            fallos += 1
            print(f"{p}: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
