"""Core ML (GPU del Mac) para el modelo completo: el lector grande y la parte densa de la maquetación como ML Programs
fp16 multifunción (producto/coreml/ del repositorio de entrenamiento). Opcional: necesita macOS 15+ y coremltools.

Adapta el proveedor de producto/coreml/proveedor_coreml.py a las interfaces de amanuensis.completo:
- LectorCTCCoreML.leer(recortes, presupuesto, sesgo) como completo.LectorCTC (recortes [alto, w16] en 0-255);
- MaquetaCoreML.mapas(lm) -> zn (sigmoide), zr, ln, tb (sigmoide), y4, h16 como el par cod_mq + maqueta en ONNX
  (la página a 960 de ancho se rellena de blanco hasta el cubo de alto; las salidas se recortan al alto real).
"""
from __future__ import annotations

import json
import os

import numpy as np


def disponible() -> bool:
    import platform

    if platform.system() != "Darwin" or os.environ.get("AMANUENSIS_COREML", "1") == "0":
        return False
    try:
        import coremltools  # noqa: F401
    except Exception:
        return False
    return True


def huella_gb() -> float:
    """Memoria física de este proceso en GB (phys_footprint, la que cuenta el vigilante; incluye la de la GPU)."""
    import ctypes
    import ctypes.util

    try:
        lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")

        class _Ri(ctypes.Structure):  # rusage_info_v2 (basta hasta ri_phys_footprint)
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
                "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
                "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
                "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
                "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten")]

        ri = _Ri()
        if lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(ri)) != 0:
            return 0.0
        return ri.ri_phys_footprint / 2**30
    except Exception:
        return 0.0


class _Funciones:
    def __init__(self, ruta, unidades="CPU_AND_GPU", max_vivas=2):
        import coremltools as ct

        self.ct, self.ruta = ct, ruta
        self.unidades = getattr(ct.ComputeUnit, unidades)
        self.max_vivas = int(os.environ.get("AMANUENSIS_COREML_FUNCIONES", max_vivas))
        self._f = {}

    def vaciar(self):
        """Suelta todas las funciones cargadas (cada una guarda su copia de los pesos); se recargan al usarlas."""
        import gc

        self._f.clear()
        gc.collect()

    def __call__(self, nombre):
        if nombre in self._f:
            self._f[nombre] = self._f.pop(nombre)  # la más reciente al final
            return self._f[nombre]
        if self._f and huella_gb() > float(os.environ.get("AMANUENSIS_COREML_LIMITE_DURO_GB", "4.6")):
            self.vaciar()  # a mitad de página: antes de cargar otra función, si la memoria ya está alta
        while len(self._f) >= self.max_vivas:  # cada función cargada guarda su copia de los pesos: memoria acotada
            import gc

            self._f.pop(next(iter(self._f)))
            gc.collect()
        comp = self.ruta[: -len(".mlpackage")] + ".mlmodelc"
        if not os.path.isdir(comp):
            m = self.ct.models.MLModel(self.ruta, function_name=nombre, compute_units=self.unidades)
            try:
                import shutil

                shutil.copytree(m.get_compiled_model_path(), comp + ".tmp")
                os.replace(comp + ".tmp", comp)
            except OSError:
                self._f[nombre] = m
                return m
        self._f[nombre] = self.ct.models.CompiledMLModel(comp, compute_units=self.unidades, function_name=nombre)
        return self._f[nombre]


class LectorCTCCoreML:
    """Mismo contrato que completo.LectorCTC. Cubos de ancho y lote fijo del paquete; máscara de columnas (el relleno
    no se ve, a diferencia del ONNX, que ve el papel de relleno del lote)."""

    def __init__(self, carpeta, unidades="CPU_AND_GPU"):
        self.cfg = json.load(open(os.path.join(carpeta, "config.json")))
        self.chars = self.cfg["chars"]
        self.alto, self.max_cols = int(self.cfg["alto"]), self.cfg.get("max_cols", 240)
        self.cubos, self.lote, self.U = sorted(self.cfg["cubos"]), int(self.cfg["lote"]), int(self.cfg["U"])
        self.f = _Funciones(os.path.join(carpeta, self.cfg["paquete"]), unidades)
        self.variante = "coreml-fp16"
        self.haz = self.corrector = None

    def leer(self, recortes, presupuesto=0, sesgo=0.0):
        from .completo import LectorCTC

        out = [None] * len(recortes)
        grupos = {}
        for i, a in enumerate(recortes):
            cols = -(-a.shape[1] // 16)
            grupos.setdefault(next((c for c in self.cubos if c >= cols), self.cubos[-1]), []).append(i)
        # primero los cubos cuya función ya está cargada (la página anterior o la lectura anterior): menos cargas de
        # funciones Core ML (0,5-1,4 s cada una); el orden no cambia la salida (máscara de columnas, lote fijo)
        for c in sorted(grupos, key=lambda c: (f"c{c}" not in self.f._f, c)):
            idx = grupos[c]
            f = self.f(f"c{c}")
            for k in range(0, len(idx), self.lote):
                trozo = idx[k: k + self.lote]
                pix = np.ones((self.lote, self.alto, 16 * c), np.float32)
                mas = np.full((self.lote, c), -1e4, np.float32)
                cols = []
                for j, i in enumerate(trozo):
                    a = recortes[i] / 255.0
                    w = min(a.shape[1], 16 * c)
                    pix[j, :, :w] = a[:, :w]
                    n = min(c, -(-a.shape[1] // 16))
                    mas[j, :n] = 0.0
                    cols.append(n)
                r = f.predict({"pixeles": pix, "mascara": mas})
                for j, i in enumerate(trozo):
                    T = cols[j] * self.U
                    out[i] = LectorCTC._decodificar(self, r["topv"][j, :T].astype(np.float32), r["topi"][j, :T].astype(np.int64), sesgo, i)
        return LectorCTC._cerrar_haz(self, out)


class MaquetaCoreML:
    def __init__(self, carpeta, unidades="CPU_AND_GPU"):
        self.cfg = json.load(open(os.path.join(carpeta, "config.json")))
        self.altos = sorted(self.cfg["altos"])
        self.f = _Funciones(os.path.join(carpeta, self.cfg["paquete"]), unidades)

    def cabe(self, H):
        return H <= self.altos[-1]

    def mapas(self, lm: np.ndarray):
        """lm [Hm, 960] en 0..1 -> (zn, zr, ln, tb, y4, h16) con forma de lote 1, como el ONNX."""
        Hm = lm.shape[0]
        Hc = next(a for a in self.altos if a >= Hm)
        g = np.ones((1, 1, Hc, 960), np.float32)
        g[0, 0, :Hm] = lm
        r = self.f(f"h{Hc}").predict({"gris": g})
        h4, h16 = Hm // 4, Hm // 16
        sig = lambda x: 1 / (1 + np.exp(-x.astype(np.float32)))
        return (sig(r["zn"][:, :, :h4]), r["zr"][:, :, :h4].astype(np.float32), r["ln"][:, :, :h4].astype(np.float32),
                sig(r["tb"][:, :, :h4]), r["y4"][:, :, :h4].astype(np.float32), r["h16"][:, :, :h16].astype(np.float32))


class CodBaseCoreML:
    """Codificador de página `cod_base` (rasgos de LineasAR) en Core ML (producto/coreml/cod_base_cm.py): una función por cubo
    de alto, ancho fijo 960; la página se rellena de blanco hasta el cubo y se recortan las filas de parches reales.
    `estados(l9)` devuelve lo mismo que cod_base_q8.onnx: [3, 1, (H/16)·(W/16), 768]."""

    def __init__(self, carpeta, unidades="CPU_AND_GPU"):
        self.cfg = json.load(open(os.path.join(carpeta, "config.json")))
        self.altos, self.ancho = sorted(self.cfg["altos"]), int(self.cfg["ancho"])
        self.esc = float(self.cfg.get("escala_estados", 64.0))
        self.f = _Funciones(os.path.join(carpeta, self.cfg["paquete"]), unidades)

    def cabe(self, l9) -> bool:
        return l9.shape[-1] == self.ancho and l9.shape[-2] <= self.altos[-1]

    def estados(self, l9: np.ndarray) -> np.ndarray:
        """l9: [1, H, 960] en 0..1."""
        H = l9.shape[1]
        Hc = next(a for a in self.altos if a >= H)
        g = np.ones((1, Hc, self.ancho), np.float32)
        g[0, :H] = l9[0]
        e = self.f(f"h{Hc}").predict({"pixeles": g})["estados"]
        est = e[:, :, : (H // 16) * (self.ancho // 16)].astype(np.float32) * self.esc
        del e
        if os.environ.get("AMANUENSIS_COREML_BASE_VIVA") != "1":  # se usa una vez por página: soltarlo libera sus pesos mientras leen el lector y la maqueta
            self.f.vaciar()
        return est
