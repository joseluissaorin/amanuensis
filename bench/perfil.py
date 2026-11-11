"""Perfil por pieza de Amanuensis en esta máquina: tiempo de cada fase de `Completo.leer` (`crono`) y de la página entera.
  con_gpu.sh python bench/perfil.py FOTO_O_CARPETA [--prensa si] [--n 3]   (AMANUENSIS_COREML=0 para CPU)
Imprime s/página y el reparto por fase (sumado) y la memoria máxima."""
import argparse
import glob
import os
import sys
import time

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("entrada")
ap.add_argument("--prensa", default="si")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--hilos", type=int, default=4)
a = ap.parse_args()
from amanuensis import completo_coreml as CM
from amanuensis.motor import Motor

fs = sorted(glob.glob(os.path.join(a.entrada, "*"))) if os.path.isdir(a.entrada) else [a.entrada]
fs = [f for f in fs if f.lower().endswith((".jpg", ".png", ".jpeg"))][: a.n]
m = Motor(prensa=a.prensa, hilos=a.hilos, verbose=False, imagenes="no")
t0 = time.time()
c = m.completo()
print(f"carga del modelo: {time.time() - t0:.1f} s  (core ml: {c.cm_maq is not None}, hilos {a.hilos})")
tot = 0.0
for k, f in enumerate(fs):
    img = Image.open(f).convert("L")
    t0 = time.time()
    p, _ = m.pagina_completa(img)
    dt = time.time() - t0
    if k:  # la primera calienta (compilación de Core ML, cachés)
        tot += dt
    print(f"{os.path.basename(f)[:40]:40s} {img.size[0]}x{img.size[1]} {len(p.lineas):4d} renglones  {dt:6.1f} s  huella {CM.huella_gb():.2f} GB", flush=True)
n = max(1, len(fs) - 1)
print(f"media sin la primera: {tot / n:.1f} s/página")
cr = c.crono
s = sum(cr.values()) or 1
for k, v in sorted(cr.items(), key=lambda x: -x[1]):
    print(f"  {k:24s} {v:7.1f} s  {100 * v / s:5.1f} %")
