"""Tubería de lectura: PDF o imagen -> páginas -> (capa de texto | líneas + lector) -> Documento.

La familia 1.0: **Amanuensis** (el modelo grande: página entera), **Amanuensis Ligero** (CPU; con --rapido usa por
dentro el lector Mínimo) y **Amanuensis Voz** (escucha.py). Internamente el modelo grande se llama «completo».
Nombres públicos de `modelo`: auto, amanuensis, ligero (alias antiguos: completo -> amanuensis; minimo -> ligero con
rapido). Expertos (ocultos en `amanu modelos`): historico, multilingue, impreso.

Elección del modelo («auto», selector por escritura y tipo de página):
- PDF con capa de texto fiable: se usa la capa (como hace Indra); el resto, OCR.
- Página escaneada: primero una sonda barata (detector Mínimo + Amanuensis Ligero, ~0,2 s). Si es impreso moderno en
  alfabeto latino (casi todas las líneas leídas con seguridad), se queda esa lectura: en impreso moderno Ligero lee
  tan bien o mejor que el lector grande y cien veces más rápido.
- Si no (manuscrito, impreso antiguo, otra escritura, maquetación compleja), Amanuensis: renglones de alta
  resolución, maquetación v4 (zonas, tablas, notas al margen, orden aprendido), recorte por la línea base y lector
  grande, con selector por escritura dentro (histórico para el alfabeto latino, multilingüe para las demás).
- Sin el modelo grande (no descargado o no publicado): la cascada de 0.4 (Ligero -> histórico -> multilingüe).
--rapido: Amanuensis Ligero en modo rápido (detector y lector Mínimos). -m amanuensis: siempre el modelo grande.
`arrojo` (0-1, 0,5 = neutro) mueve el sesgo del blanco de la CTC en todos los lectores.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import imagen as I
from . import modelos, pdf
from .modelos import entorno
from .salidas import Documento, Linea, Pagina

EXT_IMG = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".jp2", ".gif"}
UMBRAL_ML = 0.75  # confianza media por debajo de la cual se prueba el lector multilingüe
# sonda de impreso moderno latino (Ligero sobre las líneas del detector Mínimo), calibrada en el examen DEV
# (producto/completo/calibrar_sonda.py; NOTAS-producto.md §13): manda a Ligero ~20 % de las páginas por +0,15 de CER
# frente a pasarlo todo por el modelo completo (con media 0,93 y sin percentil: 28 % por +0,47)
SONDA_CONF = 0.90  # confianza media de las líneas
SONDA_P10 = 0.95  # percentil 10 de la confianza de las líneas (las flojas delatan manuscrito, gótica o maquetación rara)
SONDA_FLOJAS = 0.08  # fracción máxima de líneas con confianza < UMBRAL_ML
SONDA_MIN_LINEAS = 3
# regla por página de la maquetación de Ligero y del modo rápido (AMANUENSIS_MAQUETA_MINIMA=pagina)
REGLA_FLOJAS = 0.10  # fracción máxima de renglones flojos de la lectura clásica para quedarse con ella
REGLA_MIN_LINEAS = 3


def repo() -> Path | None:
    for c in (entorno("REPO"),):
        if c and (Path(c) / "producto" / "maqueta_lineas.py").exists() and (Path(c) / ".venv" / "bin" / "python").exists():
            return Path(c)
    return None


def _hay(nombre: str) -> bool:
    try:
        return modelos.disponible(nombre)
    except KeyError:
        return False


def figuras_de_zonas(zonas: list, umbral: float = 0.3) -> list:
    """Zonas de imagen de la maquetación v4 -> figuras en el formato de figuras.py (sin vector ni Iconclass). Las del
    mismo tipo que se solapan más del 50 % de la menor se unen (subfiguras de una misma figura)."""
    from .salidas import TIPO_FIGURA

    cand = []
    for z in zonas:
        t = TIPO_FIGURA.get(z.get("tipo") or z.get("clase"))
        if t and z.get("conf", 1) >= umbral:
            cand.append({"tipo": t, "caja": list(z["caja"]), "conf": z.get("conf", 1)})
    out = []
    for c in sorted(cand, key=lambda c: -(c["caja"][2] - c["caja"][0]) * (c["caja"][3] - c["caja"][1])):
        for o in out:
            a, b = o["caja"], c["caja"]
            inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            menor = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if o["tipo"] == c["tipo"] and menor > 0 and inter > 0.5 * menor:
                o["caja"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                break
        else:
            out.append(c)
    return sorted(out, key=lambda c: (c["caja"][1], c["caja"][0]))


def _iou(a, b):
    i = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _sin_repetir(tablas: list) -> list:
    """La maquetación puede dar la misma tabla dos veces (TableZone de SegmOnto y Table de DocLayNet): se queda una."""
    out = []
    for t in tablas:
        if not any(_iou(t["caja"], o["caja"]) > 0.7 for o in out):
            out.append(t)
    return out


def _lineas_en_tablas(p: Pagina):
    """Renglones sin zona cuyo centro cae dentro de una tabla: pasan a la zona de tabla que más la cubre (si no, se
    perderían la rejilla y las celdas y la tabla saldría como párrafos sueltos)."""
    zt = [k for k, z in enumerate(p.zonas) if (z.get("tipo") or z.get("clase")) in ("Table", "TableZone")]
    if not p.tablas:
        return
    for l in p.lineas:
        if l.zona >= 0 and l.zona_clase not in (None, "", "Text", "MainZone"):
            continue
        cx, cy = (l.caja[0] + l.caja[2]) / 2, (l.caja[1] + l.caja[3]) / 2
        t = next((t for t in p.tablas if t["caja"][0] <= cx <= t["caja"][2] and t["caja"][1] <= cy <= t["caja"][3]), None)
        if t is None:
            continue
        k = max(zt, key=lambda k: _iou(p.zonas[k]["caja"], t["caja"]), default=None)
        if k is not None and _iou(p.zonas[k]["caja"], t["caja"]) > 0.5:
            l.zona, l.zona_clase = k, p.zonas[k].get("tipo") or p.zonas[k].get("clase")
        else:
            l.zona_clase = "Table"


def tabular(cajas, minimo: int = 3) -> bool:
    """¿Hay tabla o varias columnas? Filas (misma banda vertical) con 3 o más renglones separados, al menos `minimo`
    veces: entonces la página va al modelo completo, que sabe de tablas, zonas y orden (la sonda solo lee de corrido)."""
    if len(cajas) < 2 * minimo:
        return False
    cs = sorted(cajas, key=lambda c: (c[1] + c[3]) / 2)
    filas, usado = 0, [False] * len(cs)
    for i, c in enumerate(cs):
        if usado[i]:
            continue
        yc, h = (c[1] + c[3]) / 2, c[3] - c[1]
        banda = [j for j in range(i, len(cs)) if abs((cs[j][1] + cs[j][3]) / 2 - yc) < 0.4 * h]
        for j in banda:
            usado[j] = True
        if len(banda) >= 3:
            filas += 1
    return filas >= minimo


def proveedores(dispositivo: str):
    import onnxruntime as ort

    disp = ort.get_available_providers()
    if dispositivo in ("auto", "gpu") and "CUDAExecutionProvider" in disp:
        if hasattr(ort, "preload_dlls"):  # CUDA y cuDNN de los paquetes nvidia-* de pip (o de torch), si los hay
            try:
                ort.preload_dlls()
            except Exception:
                pass
        gb = entorno("GPU_MEM_GB")  # tope de memoria de la GPU por proceso (para varios procesos en una GPU)
        if gb:
            return [("CUDAExecutionProvider", {"gpu_mem_limit": int(float(gb) * 2**30), "arena_extend_strategy": "kSameAsRequested"}),
                    "CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def con_cuda(prov) -> bool:
    return bool(prov) and any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in prov)
    return ["CPUExecutionProvider"]


class Motor:
    def __init__(self, modelo="auto", maqueta="auto", modo="auto", dpi=200, dispositivo="auto", hilos=0, verbose=True, rapido=False,
                 imagenes="auto", enderezar=None, arrojo=0.5, estructura="no", umbral_duda=None,
                 pulido=True, lexico="auto", rectificar=True):
        from .lector import sesgo_blanco
        from .salidas import UMBRAL_DUDA

        # nombres públicos de la 1.0 -> nombres internos
        modelo = {"amanuensis": "completo", "mini": "minimo", "mínimo": "minimo", "mini-base": "ligero"}.get(modelo, modelo)
        if modelo == "minimo":  # alias antiguo: Ligero en modo rápido
            rapido = True
        elif rapido and modelo in ("auto", "ligero"):  # --rapido: Ligero en modo rápido = lector Mínimo por dentro
            modelo = "minimo"

        # pulido de la página (pulido.py), corrección léxica (lexico.py: «auto» = con los léxicos de las lenguas
        # detectadas, construidos la primera vez si faltan; «local» = solo los que ya estén; «no») y renglones
        # torcidos enderezados (rectificado.py). Las variables AMANUENSIS_PULIDO, AMANUENSIS_LEXICOS y
        # AMANUENSIS_RECTIFICAR=0 los apagan también.
        self.pulido = bool(pulido) and os.environ.get("AMANUENSIS_PULIDO") != "0"
        self.lexico = "no" if os.environ.get("AMANUENSIS_LEXICOS") == "no" else (os.environ.get("AMANUENSIS_LEXICOS") or lexico)
        if self.lexico is False:
            self.lexico = "no"
        self.rectificar = bool(rectificar) and os.environ.get("AMANUENSIS_RECTIFICAR", "1") != "0"
        self.arrojo, self.sesgo = float(arrojo), sesgo_blanco(arrojo)
        self.estructura = estructura
        self.umbral_duda = UMBRAL_DUDA if umbral_duda is None else float(umbral_duda)
        self._completo = None
        self.rapido = rapido
        # enderezar (inclinación y giro de la página): por defecto sí, salvo en el modo rápido (cuesta ~0,1 s/pág.)
        self.enderezar = (not rapido) if enderezar is None else enderezar
        if rapido and maqueta == "auto":
            maqueta = "no"
        self.modelo, self.maqueta, self.modo, self.dpi = modelo, maqueta, modo, dpi
        self.prov = proveedores(dispositivo)
        self.hilos = hilos
        self.verbose = verbose
        self._lectores, self._det = {}, None
        # modelo completo (maquetación v4 y lector grande, ONNX/Core ML): «auto» y «completo»
        self.con_completo = (modelo in ("auto", "completo") and not rapido and maqueta != "no" and _hay("completo"))
        if modelo == "completo" and not self.con_completo:
            raise SystemExit("el modelo Amanuensis no está disponible (aún sin publicar: AMANUENSIS_REPO o AMANUENSIS_MODELOS)")
        # maquetación antigua del repositorio (mq_v3 por PyTorch): solo si no hay modelo completo
        # (con la maquetación mínima, Ligero ya no usa la del repositorio salvo con --maqueta si)
        self.repo = repo() if (maqueta == "si" or (maqueta == "auto" and self.maqueta_minima("ligero") is None)) and not self.con_completo else None
        if maqueta == "si" and self.repo is None and not self.con_completo:
            raise SystemExit("--maqueta si necesita el modelo Amanuensis o el repositorio eg2-ocr (AMANUENSIS_REPO)")
        # extracción de imágenes: necesita las zonas de una maquetación
        self.imagenes = (self.repo is not None or self.con_completo) and imagenes != "no"
        if imagenes == "si" and not self.imagenes:
            self.log("aviso: la extracción de imágenes necesita la maquetación del modelo Amanuensis; se omite")

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

    def completo(self):
        if self._completo is None:
            from .completo import Completo

            t0 = time.time()
            prov = self.prov if con_cuda(self.prov) else None
            self._completo = Completo(modelos.ruta("completo"), proveedores=prov, hilos=self.hilos, arrojo=self.arrojo)
            self.log(f"  modelo Amanuensis cargado en {time.time() - t0:.1f} s")
        return self._completo

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

    def _estructura(self, p: Pagina):
        if self.estructura in ("diccionario", "comic"):
            p.estructura = self.estructura
        elif self.estructura == "auto" and p.zonas:
            from . import estructura

            t = estructura.detectar_tipo(p.como_cadena())
            p.estructura = t if t != "texto" else None

    def maqueta_minima(self, para: str = "ligero"):
        """Maquetación mínima (renglones curvos, zonas, orden jerárquico y orientación propia; agente «maqueta-minima»):
        en lugar de det-mini + orientar + orden ligero. AMANUENSIS_MAQUETA_MINIMA: «pagina» (por defecto desde
        la medida del 26-10-2026, resultados/maqueta_defecto) = página a página en Ligero y su modo rápido: la clásica si
        su lectura sale firme (impreso moderno), la mínima si no (manuscrito, impreso antiguo, prensa, fotos); «auto» (el
        defecto del 19-10-2026) = la mínima siempre en Ligero y su modo rápido, pero NO en la sonda del modo automático;
        «1» = también en la sonda; «0» = nunca."""
        modo = os.environ.get("AMANUENSIS_MAQUETA_MINIMA", "pagina")
        if modo == "0" or (modo in ("auto", "pagina") and para == "sonda") or not modelos.disponible("maqueta-minima"):
            return None
        if getattr(self, "_mm", None) is None:
            from .maqueta_minima import MaquetaMinima

            self._mm = MaquetaMinima(modelos.ruta("maqueta-minima"), self.hilos, self.prov)
        return self._mm

    def orientar_pagina(self, img: Image.Image, lector: str | None = None):
        """Orientación por GEOMETRÍA (amanuensis/geometria_pagina.py; agente «orientacion», 19-10-2026): ángulo libre
        (también 10-45°), eje por la geometría de los renglones (maqueta mínima a baja resolución, o det-mini), 0°/180° por
        la asimetría tipográfica y, si no es clara, por la lectura de prueba de los renglones más largos.
        -> (imagen derecha, giro total en grados antihorarios). AMANUENSIS_ORIENTACION=clasica vuelve a la de 0.4
        (inclinación ±5° y giro por lectura con det-mini). Versión PRUDENTE desde el 19-10-2026 (ver geometria_pagina):
        los giros de 90/180/270° solo con geometría y lectura concordantes; ante la duda, 0°."""
        from . import geometria_pagina as G

        mm = self._mm_para_orientar()
        detectar, clasificador = self._detector_geom()
        lector = lector or ("ligero" if modelos.disponible("ligero") else "minimo" if modelos.disponible("minimo") else None)
        leer_puntos = None
        if lector is not None:
            from .orientacion import verosimil

            lec = self.lector(lector)

            def leer_puntos(im):
                g = np.asarray(im.convert("L"))
                if mm is not None:
                    ls = mm.analizar(Image.fromarray(g), orientar=False)["lineas"]
                    ls = sorted(ls, key=lambda l: -(l["caja"][2] - l["caja"][0]))[:8]
                    rec = [mm.recorte(g, l) for l in ls]
                else:
                    cs = sorted(self.detector().detectar(im), key=lambda c: -(c[2] - c[0]))[:8]
                    rec = [I.recorte_de_caja(im, c) for c in cs]
                if not rec:
                    return {"v": 0.0, "conf": 0.0}
                x = lec.leer(rec)
                # verosimilitud y confianza media: la versión prudente solo gira si el lector LEE en el giro propuesto
                return {"v": verosimil([y.texto for y in x], [y.conf for y in x]), "conf": float(np.mean([y.conf for y in x]))}
        leer_puntos2 = None
        if self.con_completo and os.environ.get("AMANUENSIS_ORIENTACION_GRANDE", "1") != "0" and mm is not None:
            cp = self.completo()

            def leer_puntos2(im):
                """Confianza media por carácter con el mejor de los dos lectores grandes (l3 y multilingüe)."""
                g = np.asarray(im.convert("L"))
                ls = sorted(mm.analizar(Image.fromarray(g), orientar=False)["lineas"], key=lambda l: -(l["caja"][2] - l["caja"][0]))[:8]
                rec = [mm.recorte(g, l) for l in ls]
                if not rec:
                    return 0.0
                mejor = 0.0
                for cual in ("lec", "ml"):
                    cs = [c for r in cp.leer_recortes(rec, cual) for c in r[2]]
                    if cs:
                        mejor = max(mejor, float(np.mean(cs)))
                return mejor
        if os.environ.get("AMANUENSIS_ORIENTACION") == "v1":  # la del 19-10 por la mañana (para medir; giraba de más)
            lp = (lambda im_: leer_puntos(im_)["v"]) if leer_puntos else None
            im, giro, info = G.orientar_geom_v1(img, detectar, clasificador=clasificador, leer_puntos=lp)
        else:
            im, giro, info = G.orientar_geom(img, detectar, clasificador=clasificador, leer_puntos=leer_puntos,
                                             leer_puntos2=leer_puntos2)
        self._info_orientacion = info
        return im, giro

    def _detector_geom(self):
        """-> (detectar(g_uint8) -> renglones con línea base, clasificador o None) para geometria_pagina."""
        mm = self._mm_para_orientar()
        if mm is not None:
            from .maqueta_minima import lineas_desde_mapas

            def detectar(g):
                sal, esc, _, _ = mm.mapas(g, lado=max(g.shape))
                out = []
                for caja, conf, forma, tipo in lineas_desde_mapas(sal["nuc"][0, 0], sal["lh"][0], None, mm.umbral):
                    f = 2.0 / esc
                    out.append({"caja": [v * f for v in caja], "base": [[x * f, y * f] for x, y in forma["base"]], "vertical": forma["vertical"]})
                return out

            def clasificador(g):
                sal, *_ = mm.mapas(g, lado=max(g.shape))
                return list(mm.orientacion(sal))

            return detectar, clasificador

        def detectar(g):
            cs = self.detector().detectar(Image.fromarray(g))
            return [{"caja": c[:4], "base": [[c[0], c[1] + 0.8 * (c[3] - c[1])], [c[2], c[1] + 0.8 * (c[3] - c[1])]]} for c in cs]

        return detectar, None

    def lineas_giradas(self, img: Image.Image, lineas: list, lector: str | None = None) -> list:
        """Zonas con texto girado 90°/270° sobre una página ya derecha (lomos, notas al margen en vertical, cabeceras de
        tabla): renglones que solo aparecen con la página girada y no tapan los horizontales; se leen en los dos sentidos
        y se quedan si la lectura es verosímil. Se añaden al final. AMANUENSIS_ZONAS_GIRADAS=1."""
        if os.environ.get("AMANUENSIS_ZONAS_GIRADAS", "0") != "1":
            return lineas
        from . import geometria_pagina as G
        from .orientacion import verosimil

        detectar, _ = self._detector_geom()
        g = np.asarray(img.convert("L"))
        # los renglones «verticales» de la maqueta mínima (más altos que anchos) no tapan: son justo lo que se busca aquí
        def alto(l):
            return getattr(l, "_vertical", False) or (l.caja[3] - l.caja[1]) > 1.5 * (l.caja[2] - l.caja[0])

        grupos = G.renglones_girados(g, detectar, [l.caja for l in lineas if not alto(l)])
        if not grupos:
            return lineas
        lector = lector or ("ligero" if modelos.disponible("ligero") else "minimo")
        lec = self.lector(lector)
        rot = {k: Image.fromarray(np.ascontiguousarray(np.rot90(g, k // 90))) for k in (90, 270)}
        nuevas = []
        for gr in grupos:
            cands = []
            for r in gr:
                rec = I.recorte_de_caja(rot[r["giro"]], r["caja_girada"])
                cands.append((r, rec))
                cands.append(({**r, "giro": (r["giro"] + 180) % 360}, rec.rotate(180)))
            leidas = lec.leer([c[1] for c in cands], sesgo=self.sesgo)
            puntos = [verosimil([x.texto], [x.conf]) for x in leidas]
            j = int(np.argmax(puntos))
            x = leidas[j]
            if x.conf < 0.7 or len(x.texto.strip()) < 3:
                continue
            l = Linea([round(v, 1) for v in cands[j][0]["caja"]], x.texto, round(x.conf, 4), [], confs=[round(v, 4) for v in (x.confs or [])])
            l._girado = True  # privado: no va a las salidas (asdict solo saca los campos)
            nuevas.append(l)
        if nuevas:  # fuera los renglones altos que caen sobre una zona girada (lecturas de una tira vertical sin girar)
            def tapa(c, n):
                i = max(0, min(c[2], n[2]) - max(c[0], n[0])) * max(0, min(c[3], n[3]) - max(c[1], n[1]))
                return i > 0.3 * max(1.0, (c[2] - c[0]) * (c[3] - c[1]))
            lineas = [l for l in lineas if not (alto(l) and any(tapa(l.caja, n.caja) for n in nuevas))]
        return lineas + nuevas

    def _mm_para_orientar(self):
        """La maqueta mínima como detector de la orientación (aunque no se use para leer), si está disponible."""
        if os.environ.get("AMANUENSIS_ORIENTACION") == "geom_det":  # para medir: det-mini como detector
            return None
        if getattr(self, "_mm_ori", None) is None:
            self._mm_ori = False
            try:
                if modelos.disponible("maqueta-minima"):
                    from .maqueta_minima import MaquetaMinima

                    self._mm_ori = getattr(self, "_mm", None) or MaquetaMinima(modelos.ruta("maqueta-minima"), self.hilos, self.prov)
            except Exception:
                self._mm_ori = False
        return self._mm_ori or None

    @staticmethod
    def orientacion_geom() -> bool:
        return os.environ.get("AMANUENSIS_ORIENTACION", "geom") != "clasica"

    def lineas_minima(self, img: Image.Image, cadena=None):
        """-> (imagen derecha, líneas leídas con base, polígono, tipo y zona, giro)."""
        mm = self.maqueta_minima()
        from .orientacion import verosimil

        lec0 = self.lector((cadena or ["ligero" if modelos.disponible("ligero") else "minimo"])[0] if (cadena or self.modelo in ("auto", "completo")) else self.modelo)

        def puntuar(r_):  # solo si el clasificador de orientación duda: lectura de los renglones más largos
            g_ = np.asarray(r_["imagen"])
            ls_ = sorted(r_["lineas"], key=lambda l: -(l["caja"][2] - l["caja"][0]))[:8]
            if not ls_:
                return 0.0
            x = lec0.leer([mm.recorte(g_, l) for l in ls_])
            return verosimil([y.texto for y in x], [y.conf for y in x])

        if self.enderezar and self.orientacion_geom():
            img, giro_g = self.orientar_pagina(img, lector=(cadena or [None])[0])
            r = mm.analizar(img, orientar=False)
            r["giro"] = giro_g
        else:
            # sin enderezar (modo rápido, --enderezar no) la maqueta mínima tampoco gira la página con su clasificador
            # de orientación (giraba 90/180° páginas derechas: 18 de 324 en DEV, 26-10-2026)
            r = mm.analizar(img, orientar=self.enderezar, puntuar=puntuar if self.enderezar else None)
        im = r["imagen"]
        g = np.asarray(im)
        rec = [mm.recorte(g, l) for l in r["lineas"]]
        lineas = self.leer_lineas(im, [l["caja"] for l in r["lineas"]], cadena=cadena, recortes=rec)
        canon = {"texto": "MainZone", "parrafo": "Text", "margen": "MarginTextZone", "titulo": "Title", "figura": "GraphicZone", "tabla": "TableZone",
                 "numero": "NumberingZone", "capitular": "DropCapitalZone", "cabecera": "RunningTitleZone", "pie": "Page-footer",
                 "nota": "Footnote", "formula": "Formula", "lista": "List-item", "leyenda": "Caption", "musica": "MusicZone",
                 "sello": "StampZone"}
        zonas = [{"clase": canon.get(z["tipo"], z["tipo"]), "tipo": canon.get(z["tipo"], z["tipo"]), "grueso": z["tipo"],
                  "caja": z["caja"], "conf": z["conf"]} for z in r["zonas"]]
        for l, m in zip(lineas, r["lineas"]):
            l.base, l.poligono, l.tipo = m["base"], m["poligono"], m["tipo"]
            l._vertical = bool(m.get("vertical"))
            l.zona = m["zona"]
            if m["zona"] >= 0:
                l.zona_clase = zonas[m["zona"]]["clase"]
        # orden «auto» (maqueta/minima, elegido en validación): si el lector lee con mucha seguridad (impreso moderno,
        # confianza media >= 0,95), el orden ligero sobre las cajas; si no, el de la maqueta mínima (bloques por columna)
        # + continuidad del texto (agente «orientacion»): columnas, ligero y corte XY; el de la regla se queda salvo que
        # otro le gane por más de 10 nats (impreso moderno seguro) o por cualquier margen (lo demás)
        lineas = self.reordenar(lineas, im.size, columnas=True)
        lineas = self.lineas_giradas(im, lineas, lector=(cadena or [None])[0])
        return im, lineas, r["giro"], zonas

    @staticmethod
    def orden_lm() -> bool:
        return os.environ.get("AMANUENSIS_ORDEN_LM", "1") != "0"

    def reordenar(self, lineas: list, tam, columnas: bool = False, defecto: str | None = None, margen: float | None = None,
                   xy: bool = True) -> list:
        """Orden de lectura elegido entre candidatos por la continuidad del texto (amanuensis/continuidad.py; medido en
        NOTAS-maqueta.md, «Orden por continuidad»). `lineas` llegan en el orden de su vía (columnas de la maqueta mínima,
        orden ligero del detector o el orden aprendido del modelo grande): ese es el candidato «propio»."""
        if not lineas:
            return lineas
        from . import orden as OL

        conf = float(np.mean([l.conf for l in lineas]))
        cajas = [l.caja for l in lineas]
        cands = {"propio": list(range(len(lineas)))}
        m_ = OL.modelo()
        if m_ is not None and len(lineas) <= 400:
            cands["ligero"] = OL.ordenar(m_, cajas, tam)
        from .detector import modo_orden, orden_gcn

        if modo_orden() == "gcn":  # AMANUENSIS_ORDEN=gcn: el orden por grafo pasa a ser el candidato por defecto
            o = orden_gcn(cajas, tam)
            if o is not None:
                cands["gcn"] = o
                defecto = "gcn"
        elif modo_orden() in ("ligero", "columnas") and defecto is None:
            defecto = "ligero" if modo_orden() == "ligero" and "ligero" in cands else "propio"
        if defecto is None:  # regla «auto» de la maqueta mínima: ligero si la lectura es muy segura
            defecto = "ligero" if columnas and conf >= 0.95 and "ligero" in cands else "propio"
        if not self.orden_lm():
            return [lineas[i] for i in cands[defecto]]
        from . import continuidad as C
        from .maqueta_minima import corte_xy

        if xy:
            cands["xy"] = corte_xy(cajas)
        if margen is None:
            margen = 10.0 if conf >= 0.95 else 0.0
        _, o = C.elegir_orden([l.texto for l in lineas], cands, defecto, margen)
        return [lineas[i] for i in o]

    def leer_lineas(self, img: Image.Image, cajas: list, cadena=None, recortes=None) -> list[Linea]:
        recortes = recortes if recortes is not None else [I.recorte_de_caja(img, c) for c in cajas]
        if not recortes:
            return []
        if cadena is not None:
            pass
        elif self.modelo in ("auto", "completo"):
            # CPU: Amanuensis Ligero; las líneas dudosas suben a v2b96 y, si siguen dudosas, al multilingüe
            cadena = (["ligero"] if modelos.disponible("ligero") else []) + ["historico", "multilingue"]
            if self.rapido:
                cadena = cadena[:1]
        else:
            cadena = [self.modelo]
        leidas = self.lector(cadena[0]).leer(recortes, sesgo=self.sesgo)
        if len(recortes) == len(cajas) and img is not None and self.rectificar:
            leidas, recortes = self._rectificar(img, cajas, leidas, cadena[0], recortes)
        for siguiente in cadena[1:]:
            flojas = [i for i, l in enumerate(leidas) if l.conf < UMBRAL_ML]
            if len(flojas) <= 0.1 * len(leidas):
                break
            self.log(f"  {len(flojas)} de {len(leidas)} líneas con confianza baja: releo con «{siguiente}»")
            otras = self.lector(siguiente).leer([recortes[i] for i in flojas], sesgo=self.sesgo)
            for i, o in zip(flojas, otras):
                if o.conf > leidas[i].conf + 0.05:
                    leidas[i] = o
        g = None
        if self.pulido and img is not None and any(ch in "–—" for l in leidas for ch in l.texto):
            g = np.asarray(img.convert("L"))  # para medir las rayas (signos.py)
        out = []
        for c, l in zip(cajas, leidas):
            h = c[3] - c[1]
            x0 = int(c[0] - I.MARGEN_H * h)
            out.append(Linea([round(v, 1) for v in c[:4]], l.texto, round(l.conf, 4), [round(x0 + x, 1) for x in l.xs],
                             confs=[round(v, 4) for v in (l.confs or [])]))
            if getattr(l, "lp", None) is not None and l.fotogramas is not None:  # para la corrección léxica (lexico.py)
                out[-1]._ctc = (l.lp, l.fotogramas, self.lector(cadena[0]).chars)
            if g is not None:
                from . import signos

                signos.medir_rayas(g, out[-1])  # raya media, larga o ⊢: se mide aquí y decide el documento
        return out

    def _columnas(self, img, cajas):
        """Prensa y revista a varias columnas (columnas.py): trozos de renglón unidos, cajas partidas por el canal y
        orden por columnas. AMANUENSIS_COLUMNAS=0 lo apaga."""
        if not self.pulido or os.environ.get("AMANUENSIS_COLUMNAS") == "0":
            return cajas
        from . import columnas, maqueta_yolo

        self._clases_maqueta = None
        if maqueta_yolo.activa():
            # regiones de DocLayout-YOLO (si se pide y está instalado): primero los trozos de renglón unidos y las cajas que cruzan
            # un canal partidas, luego el orden por regiones
            h = float(np.median([c[3] - c[1] for c in cajas])) if cajas else 0
            cols = columnas.comienzos(cajas, h) if cajas else []
            if len(cols) >= 2:
                cajas = columnas.unir_trozos(cajas, cols, h, np.asarray(img.convert("L")))
                cajas, _ = columnas.partir_por_canales(img, cajas)
            if os.environ.get("AMANUENSIS_MAQUETA_YOLO_ORDEN") == "1":  # (experimental: orden por regiones)
                cajas, clases, r = maqueta_yolo.maquetar(img, cajas)
                if r:
                    self._clases_maqueta = clases
                    return cajas
            else:
                if len(cols) < 2:
                    # a una columna las regiones estorban (folios, títulos y notas de un libro tomados por figuras o
                    # pies: Grice 0,56 -> 5,35 %, Pragmática 0,19 -> 2,21 %): solo el orden de siempre
                    return cajas
                cajas, rc = columnas.maquetar(img, cajas)
                clases, r = maqueta_yolo.clasificar(img, cajas)
                if r:
                    self._clases_maqueta = clases
                    self.log(f"  regiones: {r['regiones']}, {r['figuras']} figuras ({r['renglones_en_figuras']} renglones dentro, fuera de la lectura)")
                return cajas
        cajas, r = columnas.maquetar(img, cajas)
        if r:
            self.log(f"  {r['columnas']} columnas: {r['unidas']} trozos de renglón unidos, {r['partidas']} renglones partidos por el canal")
        return cajas

    def _poner_clases(self, lineas):
        """Las clases de zona de la maqueta por regiones (figura, pie, cabecera, folio, título) a los renglones leídos."""
        clases = getattr(self, "_clases_maqueta", None)
        if clases and len(clases) == len(lineas):
            for l, c in zip(lineas, clases):
                if c:
                    l.zona_clase = c
        self._clases_maqueta = None
        return lineas

    def _rectificar(self, img, cajas, leidas, lector, recortes):
        """Los renglones torcidos o curvos se releen enderezados (rectificado.py); gana la lectura más segura."""
        from . import rectificado as R

        g = img.convert("L")
        cand = []
        for i, c in enumerate(cajas):
            r, d = R.rectificar(g, c)
            if r is not None and d > R.DESVIO_MIN:
                cand.append((i, r))
        if not cand:
            return leidas, recortes
        otras = self.lector(lector).leer([r for _, r in cand], sesgo=self.sesgo)
        n = 0
        recortes = list(recortes)
        for (i, r), o in zip(cand, otras):
            if o.conf > leidas[i].conf + 0.01:
                leidas[i] = o
                recortes[i] = r
                n += 1
        if n:
            self.log(f"  {n} renglones torcidos leídos mejor enderezados")
        return leidas, recortes

    def sonda(self, img: Image.Image):
        """Detector Mínimo + Ligero (con enderezado y giro): -> (imagen derecha, cajas, líneas leídas, ¿impreso moderno?)."""
        from . import orientacion as O

        incl = giro = 0
        if self.maqueta_minima("sonda") is not None:
            img, lineas, giro, _ = self.lineas_minima(img, cadena=["ligero"] if modelos.disponible("ligero") else None)
            cajas = [l.caja for l in lineas]
        else:
            if self.enderezar and self.orientacion_geom():
                img, giro = self.orientar_pagina(img)
            elif self.enderezar:
                img, incl = O.enderezar(img)
            cajas = self.detector().detectar(img)
            if self.enderezar and not self.orientacion_geom() and modelos.disponible("ligero"):
                img, giro, cajas = O.orientar(img, cajas, self.detector().detectar, self.lector("ligero").leer)
            cajas = self._columnas(img, cajas)
            lineas = self._poner_clases(self.leer_lineas(img, cajas, cadena=["ligero"])) if modelos.disponible("ligero") else []
        ls = [l for l in lineas if l.texto.strip()]
        moderno = False
        if len(ls) >= SONDA_MIN_LINEAS and not tabular([l.caja for l in ls]):
            media = float(np.mean([l.conf for l in ls]))
            flojas = sum(l.conf < UMBRAL_ML for l in ls) / len(ls)
            p10 = float(np.percentile([l.conf for l in ls], 10))
            moderno = media >= SONDA_CONF and p10 >= SONDA_P10 and flojas <= SONDA_FLOJAS
        return img, lineas, moderno, giro, incl

    def pagina_completa(self, img: Image.Image, color: Image.Image | None = None) -> tuple:
        """Modelo completo sobre una página -> (Pagina, figuras de la maqueta en formato de figuras.py)."""
        r = self.completo().leer([img])[0]
        lineas = []
        est = r.get("estilos") or []  # por línea (cabeza tipográfica de «estructura»), si el modelo la trae
        for k, l in enumerate(r["lineas"]):
            lineas.append(Linea(l["caja"], l["texto"], l["conf"], l.get("xs", []), poligono=l.get("poligono"), base=l.get("base"),
                                zona_clase=l.get("zona_tipo"), confs=l.get("confs", []), zona=l.get("zona", -1),
                               estilos=l.get("estilos") or (est[k] if k < len(est) else []) or []))
        nombre = {"lec": "amanuensis", "ml": "amanuensis (multilingüe)"}.get(r.get("lector"), "amanuensis")
        p = Pagina(img.size[0], img.size[1], lineas, "ocr (Amanuensis, página entera)", nombre)
        p.zonas, p.tablas, p.lector = r.get("zonas", []), _sin_repetir(r.get("tablas", [])), r.get("lector", "")
        _lineas_en_tablas(p)
        if not p.tablas and len(p.lineas) >= 3:
            # orden aprendido de la cadena frente al ligero, por continuidad del texto (margen 40 nats, medido sin sellado)
            p.lineas = self.reordenar(p.lineas, img.size, defecto="propio", margen=40.0, xy=False)
        return p, figuras_de_zonas(p.zonas)

    def ocr_paginas(self, imgs: list[Image.Image], mq=None, partir: bool = True) -> list[Pagina]:
        if mq is None and self.repo is not None:
            mq = self.lineas_maqueta(imgs)
        if mq is None and partir:
            return self._ocr_dobles(imgs)
        out = []
        for k, img in enumerate(imgs):
            t0 = time.time()
            if self.con_completo:
                completo = self.modelo == "completo" or self.estructura != "no"  # la estructura rica necesita las zonas
                giro = incl = 0
                sondada = not completo
                if sondada:
                    img, lineas, moderno, giro, incl = self.sonda(img)
                    completo = not moderno
                if completo:
                    if not sondada and self.enderezar and self.orientacion_geom():
                        img, giro = self.orientar_pagina(img)
                    elif not sondada and self.enderezar:
                        from . import orientacion as O

                        img, incl = O.enderezar(img)
                    p, figs = self.pagina_completa(img)
                    p.figuras_mq = figs
                else:
                    p = Pagina(img.size[0], img.size[1], lineas, "ocr (sonda: impreso moderno latino)", "ligero")
                    p.figuras_mq = None
                if giro or incl:
                    p.fuente += (f", giro {giro}°" if giro else "") + (f", inclinación {incl}°" if incl else "")
                p.segundos = round(time.time() - t0, 2)
                p.imagen_derecha = img
                self._estructura(p)
                out.append(p)
                self.log(f"  página {k + 1}: {len(p.lineas)} líneas, {p.segundos} s ({p.fuente}; lector {p.modelo})")
                continue
            if mq is not None:
                cajas = [l["caja"] for l in mq[k]["lineas"]]
                fuente = f"ocr ({'maqueta ' + mq[k].get('tipo', '')})"
            elif self.maqueta_por_pagina():
                # regla por página (AMANUENSIS_MAQUETA_MINIMA=pagina): primero la clásica; si su lectura sale floja
                # (manuscrito, impreso antiguo, prensa degradada) o no encuentra renglones, la mínima
                p = self._pagina_clasica(img, t0)
                if not self.es_moderno(p.lineas):
                    p = self._pagina_minima(img, t0)
                    p.fuente = p.fuente[:-1] + ", por regla de página)"
                out.append(p)
                self.log(f"  página {k + 1}: {len(p.lineas)} líneas, {p.segundos} s ({p.fuente})")
                continue
            elif self.maqueta_minima("ligero") is not None:
                p = self._pagina_minima(img, t0)
                out.append(p)
                self.log(f"  página {k + 1}: {len(p.lineas)} líneas, {p.segundos} s ({p.fuente})")
                continue
            else:
                p = self._pagina_clasica(img, t0)
                out.append(p)
                self.log(f"  página {k + 1}: {len(p.lineas)} líneas, {p.segundos} s ({p.fuente})")
                continue
            lineas = self._poner_clases(self.leer_lineas(img, cajas))
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

    def maqueta_por_pagina(self) -> bool:
        """AMANUENSIS_MAQUETA_MINIMA=pagina: la maquetación se elige página a página (clásica en el impreso moderno limpio,
        mínima en lo demás). Medido en DEV y en los bancos el 26-10-2026 (agente «maqueta-defecto», COORDINACION.md)."""
        return os.environ.get("AMANUENSIS_MAQUETA_MINIMA", "pagina") == "pagina" and modelos.disponible("maqueta-minima")

    @staticmethod
    def es_moderno(lineas) -> bool:
        """¿Página para la maquetación clásica? Sí si el detector Mínimo encuentra al menos REGLA_MIN_LINEAS renglones con
        texto y, como mucho, el REGLA_FLOJAS de ellos se leen con confianza < UMBRAL_ML (lo flojo delata manuscrito,
        impreso antiguo o prensa degradada, donde gana la mínima). Umbral elegido en DEV con validación cruzada y
        comprobado en los bancos de orden (26-10-2026; resultados/maqueta_defecto)."""
        c = [l.conf for l in lineas if l.texto.strip()]
        return len(c) >= REGLA_MIN_LINEAS and sum(x < UMBRAL_ML for x in c) / len(c) <= REGLA_FLOJAS

    def _pagina_minima(self, img, t0) -> Pagina:
        """Maquetación mínima: renglones curvos, zonas, orden jerárquico + continuidad, orientación propia."""
        img, lineas, giro, zonas_mm = self.lineas_minima(img, cadena=None)
        fuente = "ocr (maqueta mínima" + (f", giro {giro}°" if giro else "") + ")"
        p = Pagina(img.size[0], img.size[1], lineas, fuente, self.modelo, round(time.time() - t0, 2))
        p.zonas = zonas_mm
        p.imagen_derecha = img
        return p

    def _pagina_clasica(self, img, t0) -> Pagina:
        """Maquetación clásica: orientación, detector Mínimo, columnas (orden por grafo en prensa) y lectura."""
        giro = incl = 0
        cadena0 = "ligero" if self.modelo == "auto" else self.modelo
        geom = self.enderezar and self.orientacion_geom()
        from . import orientacion as O

        if geom:
            img, giro = self.orientar_pagina(img, lector=cadena0 if cadena0 in ("ligero", "minimo") else None)
        elif self.enderezar:
            img, incl = O.enderezar(img)
        cajas = self.detector().detectar(img)
        if self.enderezar and not geom and cadena0 in ("ligero", "minimo"):
            img, giro, cajas = O.orientar(img, cajas, self.detector().detectar, self.lector(cadena0).leer)
        cajas = self._columnas(img, cajas)
        fuente = "ocr (detector" + (f", giro {giro}°" if giro else "") + (f", inclinación {incl}°" if incl else "") + ")"
        lineas = self._poner_clases(self.leer_lineas(img, cajas))
        # (det-mini ya da el orden ligero; la continuidad con ligero/XY aquí empeoraba: no se aplica)
        lineas = self.lineas_giradas(img, lineas, lector=cadena0 if cadena0 in ("ligero", "minimo") else None)
        return Pagina(img.size[0], img.size[1], lineas, fuente, self.modelo, round(time.time() - t0, 2))

    def _ocr_dobles(self, imgs: list[Image.Image]) -> list[Pagina]:
        """Las dobles páginas (libro abierto en un escaneo apaisado) se leen hoja a hoja y se vuelven a juntar en una
        página, con las cajas de la hoja derecha desplazadas: el orden de lectura no mezcla los renglones de las dos."""
        from . import doble

        planas, cortes = [], []
        for img in imgs:
            r = doble.partir(img)
            cortes.append(None if r is None else (r[0], img.size))
            planas += [img] if r is None else [r[1], r[2]]
        if not any(cortes):
            return self.ocr_paginas(imgs, partir=False)
        leidas, out, j = self.ocr_paginas(planas, partir=False), [], 0
        for k, c in enumerate(cortes):
            if c is None:
                out.append(leidas[j]); j += 1
                continue
            (x, (w, h)), (izq, der) = c, leidas[j: j + 2]
            j += 2
            for l in der.lineas:
                l.caja = [l.caja[0] + x, l.caja[1], l.caja[2] + x, l.caja[3]]
                l.xs = [v + x for v in l.xs]
                l.poligono = [[px + x, py] for px, py in l.poligono] if l.poligono else l.poligono
                l.base = [[px + x, py] for px, py in l.base] if l.base else l.base
            for z in der.zonas:
                if "caja" in z:
                    z["caja"] = [z["caja"][0] + x, z["caja"][1], z["caja"][2] + x, z["caja"][3]]
            nz = len(izq.zonas)
            for l in der.lineas:
                if l.zona >= 0:
                    l.zona += nz
            p = Pagina(w, h, izq.lineas + der.lineas, f"{izq.fuente}; doble página partida en x={x}", izq.modelo,
                       round(izq.segundos + der.segundos, 2), zonas=izq.zonas + der.zonas, tablas=izq.tablas + der.tablas)
            p.lector = izq.lector or der.lector
            # las figuras de la maquetación, al marco de la página entera; si una hoja se enderezó, sus coordenadas son
            # las de la hoja girada y no se puede: sin figuras
            girada = any(("giro" in q.fuente or "inclinación" in q.fuente) for q in (izq, der))
            fi, fd = getattr(izq, "figuras_mq", None), getattr(der, "figuras_mq", None)
            p.figuras_mq = None if girada or not (fi or fd) else (fi or []) + [
                dict(f, caja=[f["caja"][0] + x, f["caja"][1], f["caja"][2] + x, f["caja"][3]]) for f in (fd or [])]
            self.log(f"  página {k + 1}: doble página, partida en x={x}")
            out.append(p)
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
            from . import geometria_pagina as G

            img = G.abrir(ruta)  # con la orientación EXIF aplicada (fotos de móvil)
            color = []
            for i in range(getattr(img, "n_frames", 1)):
                img.seek(i); color.append(img.copy().convert("RGB"))
            color = color[: max_paginas or None]
            frames = [c.convert("L") for c in color]
            mq = self.lineas_maqueta(frames) if self.repo is not None else None
            d = Documento(nombre, self.ocr_paginas(frames, mq), umbral_duda=self.umbral_duda)
            if self.imagenes and (mq or self.con_completo):
                from . import figuras

                figs = [m.get("figuras") for m in mq] if mq else [getattr(p, "figuras_mq", None) for p in d.paginas]
                origs = []
                for c, p in zip(color, d.paginas):  # si la página se enderezó, se recorta de la imagen derecha
                    der = getattr(p, "imagen_derecha", None)
                    usar = c if der is None or der.size == c.size and "giro" not in p.fuente and "inclinación" not in p.fuente else der.convert("RGB")
                    origs.append(lambda u=usar: (u, 1.0))
                d.imagenes = figuras.procesar(d.paginas, figs, origs, stem)
                self._log_figuras(d)
            self._pulir(d)
            self._limpiar(d)
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
        if self.imagenes and self.repo is not None:  # la maquetación mira TODAS las páginas (también las de capa de texto) por sus figuras
            mq_todas = self.lineas_maqueta([pdf.imagen(doc[i], self.dpi, color=True) for i in range(len(paginas))])
        if pendientes:
            mq = [mq_todas[i] for i, _ in pendientes] if mq_todas else None
            for (i, _), p in zip(pendientes, self.ocr_paginas([im for _, im in pendientes], mq)):
                paginas[i] = p
        d = Documento(nombre, paginas, umbral_duda=self.umbral_duda)
        figs = [m.get("figuras") for m in mq_todas] if mq_todas else None
        if figs is None and self.imagenes and self.con_completo:  # figuras de las páginas que pasaron por el modelo completo
            figs = [getattr(p, "figuras_mq", None) for p in paginas]
        if figs and any(figs):
            from . import figuras

            DPI_FIG = 300

            def original(i):
                return lambda: (doc[i].render(scale=DPI_FIG / 72.0).to_pil().convert("RGB"), DPI_FIG / self.dpi)

            d.imagenes = figuras.procesar(paginas, figs, [original(i) for i in range(len(paginas))], stem)
            self._log_figuras(d)
        doc.close()
        self._pulir(d)
        self._limpiar(d)
        return d

    def _pulir(self, d):
        """Después de leer: diacríticos imposibles en las lenguas del documento, corrección léxica guiada por la CTC
        (lexico.py), estructura de las páginas sin maquetación (cabeceras, folios, epígrafes, figuras, filas partidas,
        basura) y correcciones de texto por series en todas (romanos, letras, ejemplos, llamadas de nota, números de
        verso y de epígrafe, personajes). Motor(pulido=False) o AMANUENSIS_PULIDO=0 lo apagan."""
        if not self.pulido:
            return
        from . import lexico, pulido

        pulido.diacriticos(d.paginas, pulido.lenguas_del_documento(d.paginas))
        from . import signos

        signos.comillas_pareadas(d.paginas)  # antes del léxico: necesita la CTC alineada con el texto
        if self.lexico != "no":
            reg = []
            n = lexico.corregir(d.paginas, registro=reg, construir_faltan=self.lexico == "auto", avisar=self.log)
            if n and os.environ.get("AMANUENSIS_DEPURAR_LEXICO"):
                for w, c, dl, v in reg:
                    print(f"  léxico: {w} -> {c} (CTC {dl:+}, total {v:+})", file=sys.stderr)
        # el orden de los bloques de columna por la palabra partida que los une (columnas.py); antes de pulir, que
        # reordena cabeceras y figuras
        from . import columnas

        lx = lexico.Lexicos(pulido.lenguas_del_documento(d.paginas) or ["es"]) if self.lexico != "no" else None
        if lx is not None and os.environ.get("AMANUENSIS_CONTINUIDAD_BLOQUES") != "0":
            for p in d.paginas:
                if not p.zonas and len(p.lineas) > 40 and "doble página" not in p.fuente:  # (dos hojas no son dos columnas)
                    columnas.por_continuidad(p, lx, lexico.RARA)
        rep = pulido.cabeceras_repetidas(d.paginas)
        for p in d.paginas:
            pulido.pulir(p, rep)
            pulido.pulir_texto(p)
            pulido.numeros_de_verso(p)
        pulido.ejemplos_numerados(d.paginas)
        pulido.numeracion_epigrafes(d.paginas)
        pulido.referencias(d.paginas)
        pulido.personajes(d.paginas)
        pulido.llamadas_de_nota(d.paginas)
        from . import signos

        signos.decidir_rayas(d.paginas)
        signos.notacion(d.paginas)

    @staticmethod
    def _limpiar(d):
        for p in d.paginas:  # atributos de trabajo que no van a las salidas
            for l in p.lineas:
                for k in ("_ctc", "_rayas"):
                    if hasattr(l, k):
                        delattr(l, k)
            for k in ("figuras_mq", "imagen_derecha"):
                if hasattr(p, k):
                    delattr(p, k)

    def _log_figuras(self, d):
        n = sum(len(getattr(p, "figuras", [])) for p in d.paginas)
        if n:
            tipos = {}
            for p in d.paginas:
                for f in p.figuras:
                    tipos[f["tipo"]] = tipos.get(f["tipo"], 0) + 1
            self.log(f"  imágenes: {n} (" + ", ".join(f"{v} {k}" for k, v in tipos.items()) + f"); {len(d.imagenes)} ficheros")
