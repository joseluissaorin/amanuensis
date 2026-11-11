"""Pesos para la GPU (CUDA): los ONNX del producto llevan los pesos en 8 bits (MatMulNBits, com.microsoft), que en la CPU
van muy bien pero en CUDA no tienen núcleo de 8 bits y caen a la CPU con copias de ida y vuelta (la GPU se queda al 7 %).
Aquí se escribe, junto a cada `*_q8*.onnx`, un `*_d32*.onnx` con los MISMOS pesos ya descuantizados ((q - zp) · escala
por bloque) en un MatMul normal: en la CPU da exactamente la misma salida (diferencia 0,0 medida), y en CUDA corre entero
en la GPU. Se hace una vez por carpeta de modelos (unos segundos por fichero) y necesita el paquete `onnx`.
MEDIDO (L40S, onnxruntime-gpu 1.22, 11-11): MatMulNBits de 8 bits SÍ corre en CUDA (ningún nodo cae a la CPU) y el
descuantizado no acelera nada (perfil 22,3 frente a 20,9 s/pág. con el mismo n-grama lento) pero cuadruplica la
memoria de los pesos en la GPU: queda APAGADO por defecto (AMANUENSIS_GPU_D32=1 lo enciende).
"""
from __future__ import annotations

import os
import sys


import re

_VAR = re.compile(r"_(q8|fp32)(?=[_.])")


def variante() -> str:
    """Pesos para CUDA: «q8» (los del producto, por defecto), «d32» (descuantizados a fp32) o «h16» (fp16, con las
    normalizaciones, potencias y softmax en fp32; solo los codificadores de 8 bits: en las redes convolucionales fp32 el
    conversor deja tipos incoherentes, Conv con fp16 y fp32, y la sesión no carga). AMANUENSIS_GPU_PESOS; AMANUENSIS_GPU_D32=1 equivale a «d32»."""
    v = os.environ.get("AMANUENSIS_GPU_PESOS", "")
    if not v and os.environ.get("AMANUENSIS_GPU_D32", "0") == "1":
        v = "d32"
    return v if v in ("d32", "h16") else "q8"


def ruta_var(ruta: str, var: str) -> str:
    d, f = os.path.split(ruta)
    return os.path.join(d, _VAR.sub("_" + var, f, count=1))


def ruta_d32(ruta: str) -> str:
    return ruta_var(ruta, "d32")


def descuantizar(src: str, dst: str) -> int:
    """MatMulNBits de 8 bits -> MatMul fp32 con los pesos descuantizados. -> número de nodos cambiados."""
    import numpy as np
    import onnx
    from onnx import helper
    from onnx import numpy_helper as nh

    m = onnx.load(src)
    g = m.graph
    ini = {i.name: i for i in g.initializer}
    nuevos, quitar, nodos, n_camb = [], set(), [], 0
    for n in g.node:
        if n.op_type != "MatMulNBits":
            nodos.append(n)
            continue
        at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        K, N, bits, bs = at["K"], at["N"], at["bits"], at["block_size"]
        if bits != 8 or n.input[1] not in ini:
            nodos.append(n)
            continue
        q = nh.to_array(ini[n.input[1]]).astype(np.float32)  # [N, bloques, bs]
        nb = q.shape[1]
        sc = nh.to_array(ini[n.input[2]]).astype(np.float32).reshape(N, nb)
        if len(n.input) > 3 and n.input[3]:
            zp = nh.to_array(ini[n.input[3]]).astype(np.float32).reshape(N, -1)[:, :nb]
            quitar.add(n.input[3])
        else:
            zp = np.full((N, nb), 128.0, np.float32)
        W = ((q - zp[:, :, None]) * sc[:, :, None]).reshape(N, nb * bs)[:, :K]
        nombre = n.input[1] + "_d32"
        nuevos.append(nh.from_array(np.ascontiguousarray(W.T, np.float32), nombre))
        quitar |= {n.input[1], n.input[2]}
        sesgo = len(n.input) >= 6 and n.input[5]
        sal = n.output[0] + "_sin_sesgo" if sesgo else n.output[0]
        nodos.append(helper.make_node("MatMul", [n.input[0], nombre], [sal], name=n.name + "_d32"))
        if sesgo:
            nodos.append(helper.make_node("Add", [sal, n.input[5]], [n.output[0]], name=n.name + "_d32_sesgo"))
        n_camb += 1
    if not n_camb:
        return 0
    del g.node[:]
    g.node.extend(nodos)
    resto = [i for i in g.initializer if i.name not in quitar]
    del g.initializer[:]
    g.initializer.extend(resto + nuevos)
    tmp = dst + ".tmp"
    if sum(i.ByteSize() for i in g.initializer) > 1.8 * 2**30:  # límite de 2 GB de protobuf
        onnx.save(m, tmp, save_as_external_data=True, location=os.path.basename(dst) + ".datos")
    else:
        onnx.save(m, tmp)
    os.replace(tmp, dst)
    return n_camb


FP32_OPS = ["Pow", "ReduceMean", "Sqrt", "Div", "Softmax", "LayerNormalization", "Erf", "Cos", "Sin", "Range", "CumSum",
            "ReduceSum", "Exp", "Log", "TopK", "Resize"]


def a_fp16(src: str, dst: str) -> None:
    """Descuantiza (si hace falta) y pasa a fp16 dejando en fp32 lo que desborda (normas, potencias, softmax)."""
    import onnx
    from onnxconverter_common import float16

    tmp = dst + ".d32.tmp"
    if descuantizar(src, tmp) == 0:
        import shutil

        shutil.copyfile(src, tmp)
    m = onnx.load(tmp)
    os.remove(tmp)
    m = float16.convert_float_to_float16(m, keep_io_types=True, op_block_list=FP32_OPS)
    # los Cast a fp32 del grafo exportado (p. ej. «_to_copy») que el conversor declara fp16: coherentes con su tipo
    tipos = {v.name: v.type.tensor_type.elem_type for v in list(m.graph.value_info) + list(m.graph.output)}
    for n in m.graph.node:
        if n.op_type == "Cast" and tipos.get(n.output[0]) == onnx.TensorProto.FLOAT16:
            for at in n.attribute:
                if at.name == "to" and at.i == onnx.TensorProto.FLOAT:
                    at.i = onnx.TensorProto.FLOAT16
    onnx.save(m, dst + ".tmp")
    os.replace(dst + ".tmp", dst)


def preparar(carpeta: str, log=None) -> int:
    """Escribe los ficheros de la variante de CUDA elegida (variante()) que falten bajo `carpeta`. -> ficheros nuevos.
    Con «q8» no hace nada. Sin `onnx` (y `onnxconverter-common` para «h16»), avisa y se queda con los de 8 bits."""
    var = variante()
    if var == "q8":
        return 0
    aviso = log or (lambda s_: print(s_, file=sys.stderr))
    pend = []
    for raiz, _, fs in os.walk(carpeta):
        for f in fs:
            r = os.path.join(raiz, f)
            if f.endswith(".onnx") and _VAR.search(f) and "_q8" in f and not os.path.exists(ruta_var(r, var)):
                pend.append(r)
    if not pend:
        return 0
    try:
        import onnx  # noqa: F401

        if var == "h16":
            import onnxconverter_common  # noqa: F401
    except ImportError:
        aviso(f"  aviso: para los pesos «{var}» de la GPU hace falta `pip install onnx onnxconverter-common`; se usan los de 8 bits")
        return 0
    hechos = 0
    for f in pend:
        try:
            if var == "h16":
                a_fp16(f, ruta_var(f, var))
                hechos += 1
            elif descuantizar(f, ruta_var(f, var)):
                hechos += 1
        except Exception as e:  # noqa: BLE001 (se queda el original)
            aviso(f"  aviso: no se pudo preparar {f} para la GPU ({var}): {e}")
    if hechos and log:
        log(f"  pesos «{var}» de la GPU preparados ({hechos} ficheros, una vez)")
    return hechos


def elegir(ruta: str, proveedores) -> str:
    """Con CUDA y una variante distinta de «q8», el fichero hermano de esa variante si existe; si no, la ruta tal cual."""
    if not proveedores or not any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in proveedores):
        return ruta
    var = variante()
    if var == "q8" or not _VAR.search(os.path.basename(ruta)):
        return ruta
    alt = ruta_var(ruta, var)
    return alt if os.path.exists(alt) else ruta


class SesionGPU:
    """Envoltorio de una sesión de onnxruntime con CUDA para varios procesos en la misma GPU:
    - tras cada ejecución la arena de la GPU se encoge a lo que está en uso (`memory.enable_memory_arena_shrinkage`):
      sin esto un solo proceso se quedaba con 24 GB de una L40S tras 52 páginas (formas variables) y cuatro no cabían;
    - si aun así falta memoria (otro proceso está en su pico), espera un poco y lo vuelve a intentar.
    AMANUENSIS_GPU_ENCOGER=0 lo apaga (una sola sesión por GPU, lo más rápido)."""

    def __init__(self, s):
        import onnxruntime as ort

        self._s = s
        self._ro = ort.RunOptions()
        if os.environ.get("AMANUENSIS_GPU_ENCOGER", "1") != "0":
            self._ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")

    def run(self, salidas, entradas, ro=None):
        import random
        import time

        for k in range(40):
            try:
                return self._s.run(salidas, entradas, ro or self._ro)
            except Exception as e:  # noqa: BLE001
                if k == 39 or not es_falta_de_memoria(e):
                    raise
                time.sleep(0.2 + random.random() * min(4.0, 0.2 * (k + 1)))

    def __getattr__(self, k):
        return getattr(self._s, k)


def es_falta_de_memoria(e) -> bool:
    m = str(e)
    return "bfc_arena" in m or "Failed to allocate" in m or "out of memory" in m.lower() or "cudaErrorMemoryAllocation" in m \
        or ("CUDA failure 2" in m) or ("cuda_call.cc" in m and "initialization" in m)


def crear_sesion(fabricar, proveedores):
    """Crea la sesión con `fabricar()`; con CUDA, si falta memoria al crearla (otro proceso en su pico), espera y reintenta."""
    if not proveedores or not any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in proveedores):
        return fabricar()
    import random
    import time

    for k in range(30):
        try:
            return SesionGPU(fabricar())
        except Exception as e:  # noqa: BLE001
            if k == 29 or not es_falta_de_memoria(e):
                raise
            time.sleep(0.5 + random.random() * min(5.0, 0.3 * (k + 1)))


def envolver(s, proveedores):
    if proveedores and any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in proveedores):
        return SesionGPU(s)
    return s
