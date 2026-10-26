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


class _Funciones:
    def __init__(self, ruta, unidades="CPU_AND_GPU", max_vivas=2):
        import coremltools as ct

        self.ct, self.ruta = ct, ruta
        self.unidades = getattr(ct.ComputeUnit, unidades)
        self.max_vivas = int(os.environ.get("AMANUENSIS_COREML_FUNCIONES", max_vivas))
        self._f = {}

    def __call__(self, nombre):
        if nombre in self._f:
            self._f[nombre] = self._f.pop(nombre)  # la más reciente al final
            return self._f[nombre]
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

    def leer(self, recortes, presupuesto=0, sesgo=0.0):
        from .completo import LectorCTC

        out = [None] * len(recortes)
        grupos = {}
        for i, a in enumerate(recortes):
            cols = -(-a.shape[1] // 16)
            grupos.setdefault(next((c for c in self.cubos if c >= cols), self.cubos[-1]), []).append(i)
        for c, idx in grupos.items():
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
                    out[i] = LectorCTC._decodificar(self, r["topv"][j, :T].astype(np.float32), r["topi"][j, :T].astype(np.int64), sesgo)
        return out


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
