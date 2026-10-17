"""Voz a texto con tiempos por palabra (codificador de audio de EmbeddingGemma 2 + cabeza CTC).

Contrato ONNX de stt/exportar.py: stt_completo.onnx (onda [B,N] 16 kHz, largos [B]) -> plano, rico,
idioma, mascara_salida; tokens_plano.json / tokens_rico.json («▁» = empieza palabra, 0 = blanco);
stt.json (ms_por_marco_salida, idiomas). Enchufe para voz2: el modelo «voz» del catálogo (o
AMANUENSIS_VOZ=/carpeta) con los mismos ficheros; para pasar a los pesos nuevos basta cambiar la carpeta.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

from .modelos import entorno

SR = 16000
VENTANA_S, SOLAPE_S = 30.0, 2.0


def cargar_audio(ruta: str) -> np.ndarray:
    """Cualquier audio o vídeo -> mono 16 kHz float32 (ffmpeg; en macOS, afconvert de reserva)."""
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "a.wav")
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", ruta, "-ac", "1", "-ar", str(SR), "-f", "wav", wav], check=True)
        elif shutil.which("afconvert"):
            subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", ruta, wav], check=True)
        else:
            wav = ruta
        with wave.open(wav) as w:
            if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise SystemExit("hace falta ffmpeg para convertir este audio (16 kHz mono)")
            a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    return a


class Escucha:
    def __init__(self, carpeta: str, hilos: int = 0, proveedores=None):
        from .lector import sesion

        self.cfg = json.load(open(os.path.join(carpeta, "stt.json")))
        fich = "stt_completo_int8.onnx" if os.path.exists(os.path.join(carpeta, "stt_completo_int8.onnx")) and not entorno("VOZ_FP32") else "stt_completo.onnx"
        self.s = sesion(os.path.join(carpeta, fich), hilos, proveedores)
        self.salidas = [o.name for o in self.s.get_outputs()]
        self.rico = "rico" in self.salidas and os.path.exists(os.path.join(carpeta, "tokens_rico.json"))
        self.tok = json.load(open(os.path.join(carpeta, "tokens_rico.json" if self.rico else "tokens_plano.json")))
        self.ms = float(self.cfg.get("ms_por_marco_salida", 40.0))
        self.idiomas = self.cfg.get("idiomas", [])

    def _ventana(self, a: np.ndarray):
        r = dict(zip(self.salidas, self.s.run(None, {"onda": a[None], "largos": np.array([len(a)], np.int64)})))
        lp = r["rico" if self.rico else "plano"][0]
        m = r.get("mascara_salida")
        t = int(m[0].sum()) if m is not None else lp.shape[0]
        idi = r.get("idioma")
        return lp[:t], (idi[0] if idi is not None else None)

    def transcribir(self, ruta: str) -> dict:
        a = cargar_audio(ruta)
        dur = len(a) / SR
        paso = VENTANA_S - 2 * SOLAPE_S
        ids, conf, tiempos, idis = [], [], [], []
        ini = 0.0
        while True:
            t0 = max(0.0, ini - SOLAPE_S)
            trozo = a[int(t0 * SR): int((ini + paso + SOLAPE_S) * SR)]
            if len(trozo) < SR // 10:
                break
            lp, idi = self._ventana(trozo)
            if idi is not None:
                idis.append(idi)
            z = lp - lp.max(-1, keepdims=True)
            p = 1.0 / np.exp(z).sum(-1)
            am = lp.argmax(-1)
            for f in range(len(am)):
                ts = t0 + f * self.ms / 1000
                if ts < ini or (ts >= ini + paso and ini + paso < dur):
                    continue  # cada instante, de la ventana donde está más centrado
                ids.append(int(am[f])); conf.append(float(p[f])); tiempos.append(ts)
            ini += paso
            if ini >= dur:
                break
        # CTC voraz con tiempos: palabra = desde su primera pieza hasta la última
        palabras, prev, act = [], 0, None
        for i, t, c in zip(ids, tiempos, conf):
            if i != 0 and i != prev:
                pieza = self.tok[i] if i < len(self.tok) else ""
                nueva = pieza.startswith("▁") or act is None
                txt = pieza.replace("▁", "")
                if nueva:
                    if act and act["texto"]:
                        palabras.append(act)
                    act = {"texto": txt, "inicio": round(t, 3), "fin": round(t + self.ms / 1000, 3), "conf": [c]}
                else:
                    act["texto"] += txt; act["fin"] = round(t + self.ms / 1000, 3); act["conf"].append(c)
            elif i != 0 and act is not None:
                act["fin"] = round(t + self.ms / 1000, 3)
            prev = i
        if act and act["texto"]:
            palabras.append(act)
        for w in palabras:
            w["conf"] = round(float(np.mean(w["conf"])), 3)
        idioma = None
        if idis and self.idiomas:
            idioma = self.idiomas[int(np.mean(idis, 0).argmax())]
        return {"fichero": os.path.basename(ruta), "duracion": round(dur, 2), "idioma": idioma,
                "texto": " ".join(w["texto"] for w in palabras), "palabras": palabras}


def _hms(t: float, sep=","):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)):03d}"


def segmentos(res: dict, max_palabras=12, max_s=6.0):
    seg, act = [], []
    for w in res["palabras"]:
        if act and (len(act) >= max_palabras or w["fin"] - act[0]["inicio"] > max_s or w["inicio"] - act[-1]["fin"] > 0.8):
            seg.append(act); act = []
        act.append(w)
        if w["texto"][-1:] in ".?!":
            seg.append(act); act = []
    if act:
        seg.append(act)
    return [(s[0]["inicio"], s[-1]["fin"], " ".join(w["texto"] for w in s)) for s in seg]


def exportar(res: dict, formato: str) -> str:
    if formato == "txt":
        return res["texto"] + "\n"
    if formato == "json":
        return json.dumps(res, ensure_ascii=False, indent=1)
    if formato == "srt":
        return "\n".join(f"{i+1}\n{_hms(a)} --> {_hms(b)}\n{t}\n" for i, (a, b, t) in enumerate(segmentos(res)))
    if formato == "vtt":
        return "WEBVTT\n\n" + "\n".join(f"{_hms(a, '.')} --> {_hms(b, '.')}\n{t}\n" for a, b, t in segmentos(res))
    raise SystemExit(f"formato de voz desconocido: {formato} (txt, srt, vtt, json)")
