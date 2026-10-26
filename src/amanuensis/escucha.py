"""Voz a texto con tiempos por palabra (codificador de audio de EmbeddingGemma 2 + cabeza CTC; voz2 desde 1.0).

Contrato ONNX de stt/exportar.py: stt_completo[_int8].onnx (onda [B,N] 16 kHz, largos [B]) -> plano, rico,
idioma, mascara_salida; tokens_plano.json / tokens_rico.json («▁» = empieza palabra, 0 = blanco);
stt.json (ms_por_marco_salida, idiomas). Modelo «voz» del catálogo o AMANUENSIS_VOZ=/carpeta.
Audio largo: ventanas de 30 s con 2 s de solape; cada ficha se toma de la ventana donde está más centrada.
Con kenlm y pyctcdecode (extra «voz-lm»), búsqueda en haz con el KenLM de la lengua sobre la salida plana y
mayúsculas y puntuación de la salida rica (lm.py); sin ellos, voraz sobre la salida rica, como antes.
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
    """lm: «auto» (por defecto; KenLM si está instalado y la lengua es una de las principales: es, ca, gl, eu, pt),
    «si» (cualquier lengua con LM) o «no» (voraz). lengua: código ISO 639-1; si no se da, la detecta el modelo.
    También por entorno: AMANUENSIS_VOZ_LM y AMANUENSIS_VOZ_LENGUA."""

    def __init__(self, carpeta: str, hilos: int = 0, proveedores=None, lm: str | None = None, lengua: str | None = None,
                 arrojo: float = 0.5):
        from .lector import sesgo_blanco, sesion

        # arrojo (0-1, 0,5 = neutro): sesgo del blanco de la CTC, como en los lectores (más arrojo = menos palabras perdidas)
        self.sesgo = sesgo_blanco(arrojo)

        self.cfg = json.load(open(os.path.join(carpeta, "stt.json")))
        fich = "stt_completo_int8.onnx" if os.path.exists(os.path.join(carpeta, "stt_completo_int8.onnx")) and not entorno("VOZ_FP32") else "stt_completo.onnx"
        self.s = sesion(os.path.join(carpeta, fich), hilos, proveedores)
        self.salidas = [o.name for o in self.s.get_outputs()]
        self.rico = "rico" in self.salidas and os.path.exists(os.path.join(carpeta, "tokens_rico.json"))
        self.tok = json.load(open(os.path.join(carpeta, "tokens_rico.json" if self.rico else "tokens_plano.json")))
        self.tok_plano = json.load(open(os.path.join(carpeta, "tokens_plano.json")))
        self.ms = float(self.cfg.get("ms_por_marco_salida", 40.0))
        self.idiomas = self.cfg.get("idiomas", [])
        self.lm = (lm or entorno("VOZ_LM", "auto")).lower()
        self.lengua = lengua or entorno("VOZ_LENGUA")

    def _ventana(self, a: np.ndarray):
        r = dict(zip(self.salidas, self.s.run(None, {"onda": a[None], "largos": np.array([len(a)], np.int64)})))
        lp = r["rico" if self.rico else "plano"][0]
        m = r.get("mascara_salida")
        t = int(m[0].sum()) if m is not None else lp.shape[0]
        idi = r.get("idioma")
        pl = r["plano"][0][:t] if "plano" in r else lp[:t]
        lp = lp[:t]
        if self.sesgo:
            b = int(self.cfg.get("blanco", 0))
            lp = lp.copy(); lp[:, b] += self.sesgo
            pl = pl.copy(); pl[:, b] += self.sesgo
        return lp, (idi[0] if idi is not None else None), pl

    def _decodificador(self, lengua):
        if self.lm in ("no", "0", "voraz") or not lengua:
            return None
        from . import lm as LM

        if not LM.disponible():
            if self.lm == "si":
                print("aviso: sin kenlm/pyctcdecode (pip install \"amanuensis[voz-lm]\"); búsqueda voraz", file=sys.stderr)
            return None
        if self.lm == "auto" and lengua not in LM.PRINCIPALES:
            return None
        try:
            return LM.decodificador(self.tok_plano, lengua)
        except Exception as ex:  # sin red, sin catálogo...: voraz
            print(f"aviso: modelo de lengua de «{lengua}» no disponible ({ex}); búsqueda voraz", file=sys.stderr)
            return None

    @staticmethod
    def _palabras_voraz(ids, tiempos, conf, tok, ms):
        """CTC voraz con tiempos: palabra = desde su primera pieza hasta la última (picos de la CTC)."""
        palabras, prev, act = [], 0, None
        for i, t, c in zip(ids, tiempos, conf):
            if i != 0 and i != prev:
                pieza = tok[i] if i < len(tok) else ""
                nueva = pieza.startswith("▁") or act is None
                txt = pieza.replace("▁", "")
                if nueva:
                    if act and act["texto"]:
                        palabras.append(act)
                    act = {"texto": txt, "inicio": round(t, 3), "fin": round(t + ms / 1000, 3), "conf": [c]}
                else:
                    act["texto"] += txt; act["fin"] = round(t + ms / 1000, 3); act["conf"].append(c)
            elif i != 0 and act is not None:
                act["fin"] = round(t + ms / 1000, 3)
            prev = i
        if act and act["texto"]:
            palabras.append(act)
        for w in palabras:
            w["conf"] = round(float(np.mean(w["conf"])), 3)
        return palabras

    def transcribir(self, ruta: str) -> dict:
        a = cargar_audio(ruta) if isinstance(ruta, str) else np.asarray(ruta, np.float32)
        dur = len(a) / SR
        paso = VENTANA_S - 2 * SOLAPE_S
        ids, conf, tiempos, idis = [], [], [], []
        lm_pal, buf, buf_desde = [], [], 0  # haz por tramos según llegan las ventanas: memoria acotada en audios de horas
        dec, lengua, decidido = None, self.lengua, False
        from . import lm as LM

        def vaciar(todo: bool):
            nonlocal buf, buf_desde
            if not buf:
                return
            x = np.concatenate(buf, 0) if len(buf) > 1 else buf[0]
            for desde, tramo in _tramos(x):
                if not todo and desde + len(tramo) == len(x) and len(tramo) < 1500:
                    buf, buf_desde = [tramo], buf_desde + desde  # el resto espera a la siguiente ventana
                    return
                g = buf_desde + desde
                for w, s0, e0 in LM.haz(dec, tramo):
                    s1, e1 = g + s0, min(g + e0, len(tiempos))
                    lm_pal.append({"texto": w, "inicio": round(tiempos[s1], 3), "fin": round(tiempos[e1 - 1] + self.ms / 1000, 3),
                                   "conf": round(float(np.mean(conf[s1:e1] or [0.0])), 3)})
            buf, buf_desde = [], buf_desde + len(x)

        ini = 0.0
        while True:
            t0 = max(0.0, ini - SOLAPE_S)
            trozo = a[int(t0 * SR): int((ini + paso + SOLAPE_S) * SR)]
            if len(trozo) < SR // 10:
                break
            lp, idi, pl = self._ventana(trozo)
            if idi is not None:
                idis.append(idi)
            if not decidido:  # la lengua del modelo de lengua: la dada o la de la primera ventana
                decidido = True
                if not lengua and idi is not None and self.idiomas:
                    lengua = self.idiomas[int(np.asarray(idi).argmax())]
                dec = self._decodificador(lengua)
            z = lp - lp.max(-1, keepdims=True)
            p = 1.0 / np.exp(z).sum(-1)
            am = lp.argmax(-1)
            sel = []
            for f in range(len(am)):
                ts = t0 + f * self.ms / 1000
                if ts < ini or (ts >= ini + paso and ini + paso < dur):
                    continue  # cada instante, de la ventana donde está más centrado
                ids.append(int(am[f])); conf.append(float(p[f])); tiempos.append(ts); sel.append(f)
            if sel and dec is not None:
                x = pl[sel[0]: sel[-1] + 1].astype(np.float32)
                x = x - x.max(-1, keepdims=True)
                buf.append(x - np.log(np.exp(x).sum(-1, keepdims=True)))  # log-softmax
                vaciar(False)
            ini += paso
            if ini >= dur:
                break
        palabras = self._palabras_voraz(ids, tiempos, conf, self.tok, self.ms)
        idioma = None
        if idis and self.idiomas:
            idioma = self.idiomas[int(np.mean(idis, 0).argmax())]
        con_lm = dec is not None
        if con_lm:
            vaciar(True)
            palabras = LM.fusionar(lm_pal, palabras)
        return {"fichero": os.path.basename(ruta) if isinstance(ruta, str) else "", "duracion": round(dur, 2), "idioma": idioma,
                "lengua": lengua or idioma, "modelo_lengua": con_lm,
                "texto": " ".join(w["texto"] for w in palabras), "palabras": palabras}


def _tramos(x: np.ndarray, max_marcos: int = 1500):
    """Parte una matriz [T, V] larga en tramos de ≤ max_marcos cortando en silencios (racha de blancos), para no
    pasar al haz matrices de varios GB en audios de horas. Devuelve (desplazamiento, tramo)."""
    i, T = 0, len(x)
    blanco = x.argmax(-1) == 0
    while i < T:
        if T - i <= max_marcos:
            yield i, x[i:]
            return
        corte = i + max_marcos
        for c in range(i + max_marcos, i + max_marcos // 2, -1):  # la racha de 4 blancos más tardía
            if blanco[c - 4: c].all():
                corte = c - 2
                break
        yield i, x[i:corte]
        i = corte


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
