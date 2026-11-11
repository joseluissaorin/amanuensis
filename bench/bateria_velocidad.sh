#!/bin/bash
# Batería de velocidad en Vast (1 GPU con CPU abundante): MISMAS 52 páginas (datos/bateria/img: 50 de val de GRESEL + 2 de
# La Vanguardia no vistas). Tiempos en resultados/bateria/tiempos.txt (una línea por prueba: etiqueta, segundos, páginas) y las
# salidas de cada sistema en resultados/bateria/<sistema>/ para medir la CER en el Mac (resumen.py / evaluar.py).
#   Amanuensis CPU (1 proceso y 4 procesos en paralelo), Amanuensis GPU (ONNX CUDA), Tesseract, y con vLLM: dots.ocr, olmOCR 2 y
#   Chandra 2 (latencia: 1 petición a la vez en 6 páginas; rendimiento: 32 a la vez en las 52).
# PP-OCRv6 no se repite aquí (en CPU, ~10 min por plana a resolución nativa: ver examen/gresel_paddle.sh).
#   colab/vast_lanzar.sh producto-bateria --gpu L40S --datos amanu_cache,bateria,prensa_lm --subir producto/amanuensis \
#     --traer resultados/bateria -- bash producto/amanuensis/bench/bateria_velocidad.sh
set -u
PASOS=${PASOS:-"cpu cpu4 gpu tess vllm"}   # PASOS="gpu vllm" relanza solo lo que faltaba
si() { [[ " $PASOS " == *" $1 "* ]]; }
cd /workspace/eg2-ocr
O=resultados/bateria; mkdir -p $O
T=$O/tiempos.txt; : > $T
IMG=datos/bateria/img
N=$(ls $IMG | wc -l)
NUC=$(python3 -c "
import os
try:
    q, p = open('/sys/fs/cgroup/cpu.max').read().split()
    print(max(2, int(int(q) / int(p))) if q != 'max' else os.cpu_count())
except Exception:
    print(os.cpu_count())")
echo "núcleos $NUC, $N páginas; $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a $T
cron() { local et=$1 n=$2; shift 2; local t0=$(date +%s.%N); "$@"; local r=$?; printf "%s\t%.1f s\t%s págs.\tcódigo %s\n" "$et" "$(echo "$(date +%s.%N) - $t0" | bc)" "$n" "$r" | tee -a $T; }
export AMANUENSIS_MODELOS=/workspace/eg2-ocr/datos/amanu_cache AMANUENSIS_RUTA_PRENSA_LM=/workspace/eg2-ocr/datos/prensa_lm
apt-get update -qq > /dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq bc tesseract-ocr tesseract-ocr-spa libgl1 libglib2.0-0 > $O/apt.log 2>&1
pip install -q ./producto/amanuensis > $O/pip.log 2>&1 || { echo "FALLO pip"; tail -5 $O/pip.log; exit 1; }
A="--prensa si --imagenes no -f json,txt -q"

# 1. Amanuensis en CPU: un proceso con todos los núcleos, y 4 procesos
si cpu && cron "amanuensis_cpu_1proc_${NUC}hilos" $N env OMP_NUM_THREADS=$NUC amanu $IMG -o $O/amanu_cpu $A --dispositivo cpu --hilos $NUC > $O/amanu_cpu.log 2>&1
rm -rf /tmp/shard; for k in 0 1 2 3; do mkdir -p /tmp/shard/$k; ls $IMG | awk -v k=$k 'NR%4==k' | while read f; do ln -s $PWD/$IMG/$f /tmp/shard/$k/$f; done; done
h4=$(( NUC / 4 )); [ $h4 -lt 2 ] && h4=2
par4() { for k in 0 1 2 3; do OMP_NUM_THREADS=$h4 amanu /tmp/shard/$k -o $O/amanu_cpu4 $A --dispositivo cpu --hilos $h4 > $O/amanu_cpu4_$k.log 2>&1 & done; wait; }
si cpu4 && cron "amanuensis_cpu_4procs_${h4}hilos" $N par4
# 2. Amanuensis en GPU (ONNX CUDA). La primera vez cayó a CPU: onnxruntime-gpu 1.2x+ se compila para CUDA 13 y la imagen trae CUDA 12.6
#    -> se fija la rama CUDA 12 (1.22.0), que usa las librerías de la imagen. Perfil por pieza con bench/perfil.py.
if si gpu; then
  pip uninstall -y -q onnxruntime onnxruntime-gpu > /dev/null 2>&1; pip install -q "onnxruntime-gpu==1.22.0" > $O/pip_ortgpu.log 2>&1
  python3 - <<'PYEOF' 2>&1 | tee -a $T
import onnxruntime as o, numpy as np
print("proveedores", o.get_available_providers())
s = o.InferenceSession("/workspace/eg2-ocr/datos/amanu_cache/amanuensis-1/lineas_fp32.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("sesión usa", s.get_providers())
PYEOF
  cron "amanuensis_gpu_cuda_1proc" $N env OMP_NUM_THREADS=8 amanu $IMG -o $O/amanu_gpu $A --dispositivo gpu --hilos 8 > $O/amanu_gpu.log 2>&1
  OMP_NUM_THREADS=8 python3 producto/amanuensis/bench/perfil.py $IMG --n 6 --hilos 8 > $O/perfil_gpu.txt 2>&1; cat $O/perfil_gpu.txt | tee -a $T
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv | tee -a $T
fi
# 3. Tesseract (spa, 1 página a la vez; los hilos de OpenMP que dé por defecto)
mkdir -p $O/tesseract
tess() { for f in $IMG/*.jpg; do s=$(basename $f .jpg); tesseract $f $O/tesseract/$s -l spa > /dev/null 2>&1; done; }
si tess && cron "tesseract_spa_secuencial" $N tess
# 4. vLLM: dots.ocr, olmOCR 2 y Chandra 2 (la 1.ª vez fallaron: la máquina tenía el controlador en CUDA 12.8 y la torch actual de vLLM pide 13:
#    lanzar con VAST_MIN_CUDA=13.0)
if si vllm; then
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=${HF_HOME:-/workspace/hf}
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > $O/uv.log 2>&1
export PATH="$HOME/.local/bin:$PATH"; VV=/workspace/vv
[ -x $VV/bin/vllm ] || uv venv -q $VV --python 3.12 && uv pip install -q --python $VV/bin/python vllm openai pillow > $O/pip_vllm.log 2>&1
uv pip install -q --python $VV/bin/python --no-deps chandra-ocr > $O/pip_chandra.log 2>&1
uv pip install -q --python $VV/bin/python pydantic-settings python-dotenv beautifulsoup4 markdownify filetype >> $O/pip_chandra.log 2>&1
PY=$VV/bin/python; export PATH="$VV/bin:$PATH"
# el cliente lee datos/gresel/<parte>: «bateria» (52 págs.) y «bateria6» (las 6 primeras, para la latencia)
mkdir -p datos/gresel/bateria datos/gresel/bateria6
for f in $IMG/*.jpg; do ln -sf $PWD/$f datos/gresel/bateria/; done
for f in $(ls $IMG/*.jpg | head -6); do ln -sf $PWD/$f datos/gresel/bateria6/; done
servir() {
  local nombre=$1 modelo=$2; shift 2
  vllm serve "$modelo" --served-model-name "$nombre" --port 8000 --gpu-memory-utilization 0.88 "$@" > $O/vllm_$nombre.log 2>&1 &
  PID=$!
  for i in $(seq 1 180); do
    curl -sf localhost:8000/health >/dev/null 2>&1 && { echo "servidor $nombre listo en $((i * 10)) s" | tee -a $T; return 0; }
    kill -0 $PID 2>/dev/null || { echo "FALLO servidor $nombre"; return 1; }
    sleep 10
  done; return 1
}
parar() { kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 5; }
rival() {  # nombre modelo args...
  local nombre=$1; shift
  if servir "$@"; then
    cron "${nombre}_vllm_latencia_1peticion" 6 $PY producto/amanuensis/bench/gresel_cliente.py --modelo $nombre --hilos 1 --partes bateria6
    cron "${nombre}_vllm_rendimiento_32peticiones" $N $PY producto/amanuensis/bench/gresel_cliente.py --modelo $nombre --hilos 32 --partes bateria
  fi
  parar
}
  rival dots dots rednote-hilab/dots.ocr --trust-remote-code --max-model-len 32768 --limit-mm-per-prompt '{"image": 1}'
rival olmocr olmocr allenai/olmOCR-2-7B-1025 --max-model-len 16384 --limit-mm-per-prompt '{"image": 1}'
rival chandra chandra datalab-to/chandra-ocr-2 --dtype bfloat16 --max-model-len 18000 --max-num-seqs 32 --max-num-batched-tokens 4096 --enable-prefix-caching --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'
fi
mkdir -p $O/vllm; for m in dots olmocr chandra; do for p in bateria bateria6; do [ -d resultados/gresel/$m/$p ] && cp -r resultados/gresel/$m/$p $O/vllm/${m}_$p; done; done
echo FIN_BATERIA | tee -a $T
