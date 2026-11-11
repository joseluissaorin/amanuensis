#!/bin/bash
# Medida en Vast sobre val100 (datos/decod_val100/img, oficial, v1): «base» (detector de renglones original, sin haz) frente a
# «actual» (detector de prensa i50 + haz con LM; es lo que hace `--prensa si` por defecto). Salida: resultados/prod_val100/<v>/
#   colab/vast_lanzar.sh producto-val100 --gpu A40 --oferta ID --datos amanu_cache,decod_val100,prensa_lm --subir producto/amanuensis \
#     --traer resultados/prod_val100 -- bash producto/amanuensis/bench/vast_val100.sh
set -u
cd /workspace/eg2-ocr
O=resultados/prod_val100; mkdir -p $O
NUC=$(python3 -c "
import os
try:
    q, p = open('/sys/fs/cgroup/cpu.max').read().split()
    print(max(2, int(int(q) / int(p))) if q != 'max' else os.cpu_count())
except Exception:
    print(os.cpu_count())")
pip install -q ./producto/amanuensis > $O/pip.log 2>&1 || { echo "FALLO pip"; exit 1; }
H=$(( NUC / 2 )); [ $H -lt 2 ] && H=2
export AMANUENSIS_MODELOS=/workspace/eg2-ocr/datos/amanu_cache AMANUENSIS_RUTA_PRENSA_LM=/workspace/eg2-ocr/datos/prensa_lm
( t0=$(date +%s); AMANUENSIS_LINEAS_PRENSA=0 OMP_NUM_THREADS=$H amanu datos/decod_val100/img -o $O/base_val100 -f json,txt --prensa si --imagenes no --haz no --hilos $H > $O/base.log 2>&1
  echo "base: código $?, $(ls $O/base_val100/*.json | wc -l) págs. en $(( $(date +%s) - t0 )) s" ) &
( t0=$(date +%s); OMP_NUM_THREADS=$H amanu datos/decod_val100/img -o $O/actual_val100 -f json,txt --prensa si --imagenes no --haz si --hilos $H > $O/actual.log 2>&1
  echo "actual: código $?, $(ls $O/actual_val100/*.json | wc -l) págs. en $(( $(date +%s) - t0 )) s" ) &
wait
echo "FIN val100"
