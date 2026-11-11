#!/bin/bash
# Medida de punta a punta EN VAST del haz con modelo de lengua del modo prensa: validación de GRESEL (90 págs.) con
# --prensa si y --haz no / --haz si, en paralelo, sobre la misma caché de modelos (conjunto amanu_cache) y el LM de
# caracteres (conjunto prensa_lm). Salida: resultados/prod_haz/<variante>_val/ (json y txt por página).
#   colab/vast_lanzar.sh producto-haz --gpu "" --datos amanu_cache,gresel_val,prensa_lm --subir producto/amanuensis \
#     --traer resultados/prod_haz -- bash producto/amanuensis/bench/vast_haz.sh
set -u
cd /workspace/eg2-ocr
O=${SALIDA:-resultados/prod_haz}; mkdir -p $O
NUC=$(python3 -c "
import os
try:
    q, p = open('/sys/fs/cgroup/cpu.max').read().split()
    print(max(2, int(int(q) / int(p))) if q != 'max' else os.cpu_count())
except Exception:
    print(os.cpu_count())")
pip install -q ./producto/amanuensis > $O/pip.log 2>&1 || { echo "FALLO pip"; tail -20 $O/pip.log; exit 1; }
command -v amanu || { echo "sin amanu"; exit 1; }
VS=${VARIANTES_HAZ:-"no si"}; NV=$(echo $VS | wc -w); H=$(( NUC / NV )); [ $H -lt 2 ] && H=2
echo "núcleos $NUC, $H hilos por variante"
export AMANUENSIS_MODELOS=/workspace/eg2-ocr/datos/amanu_cache AMANUENSIS_RUTA_PRENSA_LM=/workspace/eg2-ocr/datos/prensa_lm
for v in $VS; do
  ( t0=$(date +%s); OMP_NUM_THREADS=$H amanu datos/gresel_val/val -o $O/haz${v}_val -f json,txt --prensa si --imagenes no --haz $v --hilos $H > $O/haz$v.log 2>&1
    echo "haz $v: código $?, $(ls $O/haz${v}_val/*.json 2>/dev/null | wc -l) págs. en $(( $(date +%s) - t0 )) s" ) &
done
wait
echo "FIN haz"
