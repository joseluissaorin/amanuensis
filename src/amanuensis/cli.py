"""Amanuensis: lee cualquier página de la historia y escribe lo que le dictan (alias: amanu, eg2ocr).

Familia: Amanuensis (el modelo grande, página entera), Amanuensis Ligero (CPU; --rapido usa el lector Mínimo) y
Amanuensis Voz (amanuensis escucha). Por defecto (-m auto) cada página va a Amanuensis o a Ligero según lo que sea.

  amanuensis documento.pdf                     # Markdown a la salida estándar
  amanuensis pagina.jpg -f tei -o salida/      # TEI en salida/pagina.tei.xml
  amanuensis carpeta/ -f page,alto,json -o x/  # varios formatos a la vez, todas las imágenes y PDF
  amanuensis folio.jpg -m amanuensis -f tei    # siempre el modelo grande: maquetación, notas al margen, tablas
  amanuensis escaneos/ -m ligero -r            # Ligero en modo rápido (lector Mínimo), décimas por página
  amanuensis fax.png --arrojo 0.8              # escaneo muy malo: el lector se arriesga más (0,5 = neutro)
  amanuensis escucha entrevista.mp3 -f srt     # voz a texto con tiempos (txt, srt, vtt, json)
  amanuensis modelos                           # catálogo y qué hay ya descargado
  amanuensis lexicos                           # léxicos de las 31 lenguas: cuáles hay; «construir [lenguas]»

Por defecto, Markdown, TXT y HTML salen como texto de lectura: sin folios, cabeceras ni marcas de página, con los
párrafos y las palabras partidas unidos también entre páginas y el verso en sus renglones (--fiel: página a página).
Después de leer se pule la página y se corrigen las palabras dudosas con los léxicos de las lenguas detectadas
(--pulido no, --lexico no|local, --rectificar no los apagan).
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
    ap.add_argument("--lm", default=None, choices=["auto", "si", "no"], help="modelo de lengua KenLM (auto: es, ca, gl, eu y pt si está instalado)")
    ap.add_argument("--lengua", default=None, help="código ISO 639-1 (si no, la detecta el modelo)")
    ap.add_argument("--arrojo", type=float, default=0.5, help="0-1; más alto = pierde menos palabras en audio malo (0,5 = neutro)")
    a = ap.parse_args(argv)
    from . import modelos
    from .modelos import entorno
    from .escucha import Escucha, exportar

    carpeta = entorno("VOZ") or modelos.ruta("voz")
    e = Escucha(carpeta, a.hilos, lm=a.lm, lengua=a.lengua, arrojo=a.arrojo)
    fs = a.formato.split(",")
    lista = list(entradas(a.audios, EXT_AUDIO))
    for p in lista:
        t0 = time.time()
        r = e.transcribir(str(p))
        print(f"{p.name}: {r['duracion']} s de audio en {time.time() - t0:.1f} s (idioma {r['idioma']})", file=sys.stderr)
        for f in fs:
            escribir(exportar(r, f), a.salida, p, "." + f, len(lista) > 1 or len(fs) > 1)


def cmd_modelos(todos=False):
    from . import modelos

    cat = modelos.catalogo()
    print(f"origen: {cat['base']}\ncaché: {modelos.cache()}")
    for n, m in cat["modelos"].items():
        if m.get("oculto") and not todos:
            continue
        d = modelos.cache() / m["carpeta"]
        hay = all((d / f).exists() for f in m["ficheros"])
        mb = sum(v["bytes"] if isinstance(v, dict) else v for v in m["ficheros"].values()) / 1e6
        estado = "descargado" if hay else "local (sin publicar)" if modelos._local(m) else ("pendiente de publicar" if m.get("pendiente") else f"se descarga al usarlo ({mb:.0f} MB)")
        nc = " [uso no comercial]" if "no comercial" in m.get("licencia", "") else ""
        print(f"  {n:<16} {estado:<34} {m['descripcion']}{nc}")
    if not todos:
        print("  (piezas internas y versiones anteriores: amanuensis modelos --todos; alias: completo = amanuensis, minimo = ligero --rapido)")


def lexicos(argv):
    """amanu lexicos: estado de los léxicos; amanu lexicos construir [lenguas | --todas]."""
    from . import lexico as X

    if argv and argv[0] == "construir":
        pedidas = [l for l in argv[1:] if l != "--todas"] or (X.LENGUAS if "--todas" in argv or len(argv) == 1 else [])
        malas = [l for l in pedidas if l not in X.LENGUAS]
        if malas:
            print(f"lenguas desconocidas: {' '.join(malas)} (valen: {' '.join(X.LENGUAS)})", file=sys.stderr)
            return 2
        for l in pedidas:
            t0 = time.time()
            try:
                r = X.construir(l)
                print(f"{l}: {os.path.getsize(r) // 1024} KB en {time.time() - t0:.0f} s -> {r}")
            except Exception as e:
                print(f"{l}: no he podido construirlo ({e})", file=sys.stderr)
        return 0
    print(f"léxicos en {X.carpeta_lexicos()} (se construyen solos la primera vez que hace falta una lengua):")
    for l in X.LENGUAS:
        r = X.ruta_lengua(l)
        nota = " (sin corrección por palabra: no separa palabras)" if l in X.SIN_ESPACIOS else ""
        print(f"  {l}  {'listo, ' + str(os.path.getsize(r) // 1024) + ' KB' if r else 'falta'}{nota}")
    print("fuentes: FrequencyWords 2018 (OpenSubtitles, CC BY-SA 4.0) y wikipedia-word-frequency (Wikipedia, CC BY-SA)")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "escucha":
        return cmd_escucha(argv[1:])
    if argv and argv[0] == "modelos":
        return cmd_modelos("--todos" in argv[1:])
    if argv and argv[0] == "lexicos":
        return lexicos(argv[1:])
    if argv and argv[0] == "visor":
        from .visor import main as visor

        return visor(argv[1:])
    ap = argparse.ArgumentParser(prog="amanuensis", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entradas", nargs="+", help="PDF, imágenes o carpetas")
    ap.add_argument("-f", "--formato", default="md", help=f"{', '.join(FORMATOS)} (varios separados por comas)")
    ap.add_argument("-o", "--salida", help="carpeta de salida (por defecto, la salida estándar)")
    ap.add_argument("-m", "--modelo", default="auto",
                    help="auto (Amanuensis o Ligero según la página), amanuensis (el modelo grande), ligero; alias antiguos: "
                         "completo = amanuensis, minimo = ligero --rapido; para expertos: historico, multilingue, impreso")
    ap.add_argument("-r", "--rapido", action="store_true", help="Amanuensis Ligero en modo rápido: detector y lector Mínimos (CPU, ~0,1 s por página)")
    ap.add_argument("--maqueta", default="auto", choices=["auto", "si", "no"], help="maquetación completa del proyecto (zonas, líneas con base, orden)")
    ap.add_argument("--pulido", default="si", choices=["si", "no"],
                    help="pulir la página tras leerla: cabeceras, folios, epígrafes, series, diacríticos imposibles")
    ap.add_argument("--lexico", default="auto", choices=["auto", "local", "no"],
                    help="corrección léxica guiada por la CTC con las lenguas detectadas (auto: construye los léxicos "
                         "que falten; local: solo los que haya)")
    ap.add_argument("--rectificar", default="si", choices=["si", "no"], help="releer enderezados los renglones curvos")
    ap.add_argument("--fiel", action="store_true",
                    help="md, txt y html página a página, con folios, cabeceras y marcas de página (por defecto se quitan "
                         "y se unen los párrafos y las palabras partidas entre páginas)")
    ap.add_argument("--modo", default="auto", choices=["auto", "ocr", "texto"], help="PDF: capa de texto si es fiable (auto), siempre OCR, o siempre la capa")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--enderezar", default="auto", choices=["auto", "si", "no"], help="corregir inclinación y giro (90/180/270) de páginas escaneadas o fotografiadas; auto: sí salvo con -r")
    ap.add_argument("--imagenes", default="auto", choices=["auto", "si", "no"], help="extraer figuras, grabados, capitulares y sellos (con la maquetación completa)")
    ap.add_argument("--paginas", type=int, default=0, help="como mucho N páginas por fichero")
    ap.add_argument("--arrojo", type=float, default=0.5,
                    help="cuánto se arriesga el lector, de 0 a 1 (0,5 = neutro): más alto en escaneos muy malos o de baja resolución "
                         "(calla menos letras), más bajo en limpio o si prefieres huecos a inventos")
    ap.add_argument("--dudas", type=float, default=0.5, metavar="P",
                    help="las letras con probabilidad menor que P se marcan como dudosas (TEI <unclear>, hOCR, ALTO, JSON); 0 = no marcar")
    ap.add_argument("--estructura", default="no", choices=["no", "auto", "diccionario", "comic"],
                    help="estructura rica con el modelo grande: entradas de diccionario o viñetas de cómic (auto: detección aún floja)")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--hilos", type=int, default=0)
    ap.add_argument("-q", "--silencio", action="store_true")
    ap.add_argument("--version", action="version", version=f"amanuensis {__version__}")
    a = ap.parse_args(argv)
    a.modelo = {"mini": "minimo", "mini-base": "ligero", "mínimo": "minimo"}.get(a.modelo, a.modelo)
    if a.modelo not in ("auto", "amanuensis", "completo", "ligero", "minimo", "historico", "multilingue", "impreso"):
        ap.error(f"modelo desconocido: {a.modelo} (auto, amanuensis, ligero)")
    fs = a.formato.split(",")
    for f in fs:
        if f not in FORMATOS:
            ap.error(f"formato desconocido: {f}")
    from .motor import Motor

    if not 0 <= a.arrojo <= 1:
        ap.error("--arrojo va de 0 a 1")
    m = Motor(a.modelo, a.maqueta, a.modo, a.dpi, a.dispositivo, a.hilos, not a.silencio, a.rapido, a.imagenes,
              {"auto": None, "si": True, "no": False}[a.enderezar], arrojo=a.arrojo, estructura=a.estructura, umbral_duda=a.dudas,
              pulido=a.pulido == "si", lexico=a.lexico, rectificar=a.rectificar == "si")
    lista = list(entradas(a.entradas, EXT_DOC))
    fallos = 0
    for p in lista:
        try:
            t0 = time.time()
            d = m.leer(str(p), a.paginas)
            d.fiel = a.fiel
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
