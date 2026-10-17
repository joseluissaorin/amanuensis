"""Tubería de lectura: PDF o imagen -> páginas -> (capa de texto | líneas + lector) -> Documento.

Elección del modelo («auto»):
- PDF con capa de texto fiable: se usa la capa (como hace Indra); el resto, OCR.
- Líneas: maquetación del proyecto (mq_v3: zonas, líneas con base y orden) si se encuentra el repositorio
  de entrenamiento (AMANUENSIS_REPO); si no, el detector Mínimo.
- Lector: «ligero» (Amanuensis Ligero, 3,5 M, CPU; en impreso moderno lee mejor que v2b96); si más del 10 % de
  las líneas salen con confianza baja, esas se releen con «historico» (v2b96) y luego con «multilingue»
  (ml_v1), y se queda, línea a línea, la lectura más segura. --rapido: solo Ligero y el detector Mínimo.
Para cambiar a los pesos v3/v4 basta con editar catalogo.json (carpeta del modelo) o AMANUENSIS_MODELOS.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

from . import imagen as I
from . import modelos, pdf
from .modelos import entorno
from .salidas import Documento, Linea, Pagina

EXT_IMG = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".jp2", ".gif"}
UMBRAL_ML = 0.75  # confianza media por debajo de la cual se prueba el lector multilingüe


def repo() -> Path | None:
    for c in (entorno("REPO"),):
        if c and (Path(c) / "producto" / "maqueta_lineas.py").exists() and (Path(c) / ".venv" / "bin" / "python").exists():
            return Path(c)
    return None


def proveedores(dispositivo: str):
    import onnxruntime as ort

    disp = ort.get_available_providers()
    if dispositivo in ("auto", "gpu") and "CUDAExecutionProvider" in disp:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class Motor:
    def __init__(self, modelo="auto", maqueta="auto", modo="auto", dpi=200, dispositivo="auto", hilos=0, verbose=True, rapido=False,
                 imagenes="auto"):
        self.rapido = rapido
        if rapido and maqueta == "auto":
            maqueta = "no"
        self.modelo, self.maqueta, self.modo, self.dpi = modelo, maqueta, modo, dpi
        self.prov = proveedores(dispositivo)
        self.hilos = hilos
        self.verbose = verbose
        self._lectores, self._det = {}, None
        self.repo = repo() if maqueta in ("auto", "si") else None
        if maqueta == "si" and self.repo is None:
            raise SystemExit("--maqueta si necesita el repositorio eg2-ocr (AMANUENSIS_REPO)")
        # extracción de imágenes: necesita las zonas de la maquetación completa
        self.imagenes = self.repo is not None and imagenes != "no"
        if imagenes == "si" and self.repo is None:
            self.log("aviso: la extracción de imágenes necesita la maquetación completa (AMANUENSIS_REPO); se omite")

    def log(self, *a):
        if self.verbose:
            print(*a, file=sys.stderr, flush=True)

    # ---------- modelos ----------
    def lector(self, nombre):
        if nombre not in self._lectores:
            from .lector import LectorEG2, LectorMini

            ruta = modelos.ruta(nombre)
            self._lectores[nombre] = LectorMini(ruta, self.hilos, self.prov) if nombre in ("ligero", "minimo") else LectorEG2(ruta, hilos=self.hilos, proveedores=self.prov)
        return self._lectores[nombre]

    def detector(self):
        if self._det is None:
            from .detector import DetectorEG2, DetectorMini

            if modelos.disponible("detector-minimo"):
                self._det = DetectorMini(modelos.ruta("detector-minimo"), self.hilos, self.prov)
            else:
                self._det = DetectorEG2(modelos.ruta("detector"), hilos=self.hilos, proveedores=self.prov)
        return self._det

    # ---------- líneas ----------
    def lineas_maqueta(self, imgs: list[Image.Image]):
        with tempfile.TemporaryDirectory() as d:
            rutas = []
            for i, im in enumerate(imgs):
                r = os.path.join(d, f"{i:04d}.png"); im.save(r); rutas.append(r)
            sal = os.path.join(d, "mq.json")
            orden = [str(self.repo / ".venv" / "bin" / "python"), str(self.repo / "producto" / "maqueta_lineas.py"), *rutas, "--salida", sal]
            if self.imagenes:
                orden.append("--figuras")
            gpu = self.repo / "herramientas" / "con_gpu.sh"
            if gpu.exists():
                orden = [str(gpu), *orden]
            r = subprocess.run(orden, cwd=self.repo, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0 or not os.path.exists(sal):
                self.log("aviso: la maquetación del proyecto falló; uso el detector de líneas\n" + r.stderr[-800:])
                return None
            return json.load(open(sal))["paginas"]

    def leer_lineas(self, img: Image.Image, cajas: list) -> list[Linea]:
        recortes = [I.recorte_de_caja(img, c) for c in cajas]
        if not recortes:
            return []
        if self.modelo == "auto":
            # CPU: Amanuensis Ligero; las líneas dudosas suben a v2b96 y, si siguen dudosas, al multilingüe
            cadena = (["ligero"] if modelos.disponible("ligero") else []) + ["historico", "multilingue"]
            if self.rapido:
                cadena = cadena[:1]
        else:
            cadena = [self.modelo]
        leidas = self.lector(cadena[0]).leer(recortes)
        for siguiente in cadena[1:]:
            flojas = [i for i, l in enumerate(leidas) if l.conf < UMBRAL_ML]
            if len(flojas) <= 0.1 * len(leidas):
                break
            self.log(f"  {len(flojas)} de {len(leidas)} líneas con confianza baja: releo con «{siguiente}»")
            otras = self.lector(siguiente).leer([recortes[i] for i in flojas])
            for i, o in zip(flojas, otras):
                if o.conf > leidas[i].conf + 0.05:
                    leidas[i] = o
        out = []
        for c, l in zip(cajas, leidas):
            h = c[3] - c[1]
            x0 = int(c[0] - I.MARGEN_H * h)
            out.append(Linea([round(v, 1) for v in c[:4]], l.texto, round(l.conf, 4), [round(x0 + x, 1) for x in l.xs]))
        return out

    def ocr_paginas(self, imgs: list[Image.Image], mq=None) -> list[Pagina]:
        if mq is None and self.repo is not None:
            mq = self.lineas_maqueta(imgs)
        out = []
        for k, img in enumerate(imgs):
            t0 = time.time()
            if mq is not None:
                cajas = [l["caja"] for l in mq[k]["lineas"]]
                fuente = f"ocr ({'maqueta ' + mq[k].get('tipo', '')})"
            else:
                cajas = self.detector().detectar(img)
                fuente = "ocr (detector)"
            lineas = self.leer_lineas(img, cajas)
            if mq is not None:
                zs = mq[k].get("zonas", [])
                for l, m in zip(lineas, mq[k]["lineas"]):
                    if m.get("poligono"):
                        l.poligono = m["poligono"]; l.base = m.get("base"); l.tipo = m.get("tipo")
                    if 0 <= m.get("zona", -1) < len(zs):
                        l.zona_clase = zs[m["zona"]]["clase"]
            p = Pagina(img.size[0], img.size[1], lineas, fuente, self.modelo, round(time.time() - t0, 2))
            if mq is not None:
                p.zonas = mq[k].get("zonas", [])
            out.append(p)
            self.log(f"  página {k + 1}: {len(lineas)} líneas, {p.segundos} s ({fuente})")
        return out

    # ---------- documentos ----------
    def leer(self, ruta: str, max_paginas: int = 0) -> Documento:
        ruta = str(ruta)
        nombre = os.path.basename(ruta)
        self.log(f"{nombre}:")
        with open(ruta, "rb") as f:
            es_pdf = b"%PDF-" in f.read(1024)
        stem = Path(ruta).stem
        if not es_pdf:
            img = Image.open(ruta)
            color = []
            for i in range(getattr(img, "n_frames", 1)):
                img.seek(i); color.append(img.copy().convert("RGB"))
            color = color[: max_paginas or None]
            frames = [c.convert("L") for c in color]
            mq = self.lineas_maqueta(frames) if self.repo is not None else None
            d = Documento(nombre, self.ocr_paginas(frames, mq))
            if self.imagenes and mq:
                from . import figuras

                d.imagenes = figuras.procesar(d.paginas, [m.get("figuras") for m in mq], [(lambda c=c: (c, 1.0)) for c in color], stem)
                self._log_figuras(d)
            return d
        doc = pdf.abrir(ruta)
        paginas: list = [None] * len(doc)
        if max_paginas:
            paginas = paginas[:max_paginas]
        pendientes = []
        for i in range(len(paginas)):
            pag = doc[i]
            if self.modo in ("auto", "texto"):
                ls = pdf.lineas_texto(pag, self.dpi)
                ok, nota = pdf.capa_fiable(ls)
                if ok or (self.modo == "texto" and ls):
                    w, h = pag.get_size()
                    paginas[i] = Pagina(int(w * self.dpi / 72), int(h * self.dpi / 72), [Linea([round(v, 1) for v in c], t) for c, t in ls], "texto")
                    self.log(f"  página {i + 1}: capa de texto ({len(ls)} líneas)")
                    continue
                self.log(f"  página {i + 1}: {nota}; OCR")
            pendientes.append((i, pdf.imagen(pag, self.dpi)))
        mq_todas = None
        if self.imagenes:  # la maquetación mira TODAS las páginas (también las de capa de texto) por sus figuras
            mq_todas = self.lineas_maqueta([pdf.imagen(doc[i], self.dpi, color=True) for i in range(len(paginas))])
        if pendientes:
            mq = [mq_todas[i] for i, _ in pendientes] if mq_todas else None
            for (i, _), p in zip(pendientes, self.ocr_paginas([im for _, im in pendientes], mq)):
                paginas[i] = p
        d = Documento(nombre, paginas)
        if mq_todas:
            from . import figuras

            DPI_FIG = 300

            def original(i):
                return lambda: (doc[i].render(scale=DPI_FIG / 72.0).to_pil().convert("RGB"), DPI_FIG / self.dpi)

            d.imagenes = figuras.procesar(paginas, [m.get("figuras") for m in mq_todas], [original(i) for i in range(len(paginas))], stem)
            self._log_figuras(d)
        doc.close()
        return d

    def _log_figuras(self, d):
        n = sum(len(getattr(p, "figuras", [])) for p in d.paginas)
        if n:
            tipos = {}
            for p in d.paginas:
                for f in p.figuras:
                    tipos[f["tipo"]] = tipos.get(f["tipo"], 0) + 1
            self.log(f"  imágenes: {n} (" + ", ".join(f"{v} {k}" for k, v in tipos.items()) + f"); {len(d.imagenes)} ficheros")
