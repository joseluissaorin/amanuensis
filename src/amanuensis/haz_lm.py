"""[Copia de decod/prensa_haz.py de eg2-ocr, integrada en el producto como opción del modo prensa: `--haz si`.]

Búsqueda en haz CTC con modelo de lengua de caracteres para el lector de Amanuensis (modo prensa).

Módulo AUTÓNOMO (solo numpy y la biblioteca estándar): se puede copiar tal cual al producto.

Entrada: lo que ya recibe `completo.LectorCTC._decodificar(v, ii, sesgo)`: las 16 mejores log-probabilidades por
fotograma (`v` [T, K]) y sus clases (`ii` [T, K]; 0 = blanco; clase i -> chars[i-1]), más la lectura voraz ya hecha
`(texto, conf, confs, fotogramas)`. Salida: la misma tupla, corregida.

    d = Decodificador("carpeta_del_modelo_de_lengua")        # una vez
    texto, conf, confs, frames = d.decodificar(v, ii, chars, voraz)

Principios (edición filológica: corregir errores de lectura SIN modernizar):
  1. Solo se abre el haz en renglones con algún fotograma dudoso (`duda`); los seguros salen como los leyó el lector.
  2. El haz propone; cada cambio respecto a la lectura voraz se acepta por separado. No se acepta un cambio que solo
     pone, quita o cambia un diacrítico («fué» -> «fue», «composicion» -> «composición»), ni uno que solo cambia
     mayúscula por minúscula, ni cambios entre cifras: eso lo decide la imagen (`solo_letras`).
  3. Región de confianza: el renglón corregido no puede ser para el lector mucho menos verosímil que el voraz
     (`delta`, en nats de la CTC exacta), ni cambiar más de `max_cambios` caracteres o `max_frac` del renglón.
"""
from __future__ import annotations

import difflib
import json
import os
import math
import unicodedata

import numpy as np

NEG = -math.inf
INICIO = "\x02"
TABLA = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "„": '"', "—": "–"})


def _lae(a: float, b: float) -> float:
    if a == NEG:
        return b
    if b == NEG:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class NGrama:
    """n-grama de caracteres compacto (formato de decod/ngrama_compacto.py; ver prensa_lm.py)."""

    def __init__(self, carpeta: str, max_cache: int = 60_000):
        m = json.load(open(f"{carpeta}/meta.json"))
        self.n, self.base, self.piso = m["n"], m["base"], m["piso"]
        self.ids = {c: i + 1 for i, c in enumerate(m["chars"])}
        self.otro = self.ids.get(m.get("otro"), 0)
        carga = lambda x: np.load(f"{carpeta}/{x}.npy")  # en RAM (63 MB): sin la sobrecarga de np.memmap por cada acceso
        self.o = [(carga(f"o{k}_ctx"), carga(f"o{k}_lam"), carga(f"o{k}_clave"), carga(f"o{k}_a")) for k in range(self.n)]
        self.cache: dict = {}
        self.max_cache = max_cache
        self._ini = INICIO * (self.n - 1)
        self._ptr = [None] * self.n

    def _hijos(self, k):
        """Para el orden k: inicio y fin de los hijos de cada contexto en o{k}_clave (una vez, vectorizado; mismo
        resultado que buscar cod·B y cod·B + B en cada consulta). Los contextos van ordenados, así que basta con dos
        searchsorted de golpe."""
        h = self._ptr[k]
        if h is None:
            cx, _, cl, _ = self.o[k]
            B = np.uint64(self.base)
            h = self._ptr[k] = (np.searchsorted(cl, cx * B).astype(np.int64), np.searchsorted(cl, cx * B + B).astype(np.int64))
        return h

    def dist(self, ctx: str) -> np.ndarray:
        """log natural de P(c | ctx) por id de carácter. ctx: texto anterior (se pliega y se recorta aquí)."""
        ctx = (self._ini + ctx[-(self.n - 1):].translate(TABLA))[-(self.n - 1):]
        r = self.cache.get(ctx)
        if r is not None:
            return r
        B = self.base
        p = np.full(B, self.piso, dtype=np.float64)
        cod = 0
        ids, otro = self.ids, self.otro
        for k in range(self.n):
            if k:
                cod += ids.get(ctx[-k], otro) * B ** (k - 1)
            cx, lam, cl, a = self.o[k]
            i = int(cx.searchsorted(np.uint64(cod)))
            if i >= len(cx) or int(cx[i]) != cod:
                break  # si falta un contexto, faltan los más largos
            ini, fin = self._hijos(k)
            i0, i1 = int(ini[i]), int(fin[i])
            p *= float(lam[i])
            p[(cl[i0:i1] - np.uint64(cod * B)).astype(np.int64)] += a[i0:i1]
        r = np.log(p).tolist()
        if len(self.cache) >= self.max_cache:
            self.cache.clear()
        self.cache[ctx] = r
        return r

    def logp(self, ctx: str, c: str) -> float:
        return self.dist(ctx)[self.ids.get(c.translate(TABLA), self.otro)]

    def frase(self, t: str, fin: bool = True) -> float:
        s = 0.0
        for i, c in enumerate(t):
            s += self.logp(t[:i], c)
        return s + (self.logp(t, "\n") if fin else 0.0)


def haz(v: np.ndarray, ii: np.ndarray, chars: list, lm: NGrama | None, alfa=0.5, beta=1.0, ancho=8, umbral=-7.0,
        nmejores=1, contexto: str = "") -> list:
    """Haz CTC por prefijos sobre las K mejores por fotograma. -> [(texto, puntos)] de mejor a peor.
    puntos = ctc + alfa·lm + beta·caracteres.
    (Velocidad, 11-11: misma aritmética y mismo orden de inserción que la versión de «lengua», con la suma logarítmica
    en línea, los candidatos y el índice del n-grama de cada clase precalculados; salida idéntica, ver velocidad/.)"""
    T, K = v.shape
    es_b = ii == 0
    hay_b = es_b.any(1)
    pb_t = np.where(hay_b, np.where(es_b, v, -np.inf).max(1), v.min(1)).tolist()
    nch = len(chars)
    # candidatos por fotograma (carácter, log-prob), solo en los fotogramas que no son de blanco seguro
    acepta = (ii != 0) & (v > umbral) & (ii <= nch)
    V, I, A = v.tolist(), ii.tolist(), acepta.tolist()
    usar_lm = lm is not None and bool(alfa)
    if usar_lm:
        ids_lm, otro = lm.ids, lm.otro
        lmi = {}
        dpref = {}  # prefijo -> distribución del n-grama (la misma que lm.dist; sin rehacer la clave en cada fotograma)
    exp, log1p = math.exp, math.log1p
    haces: dict = {"": [0.0, NEG, 0.0]}
    for t in range(T):
        pb = pb_t[t]
        if pb > -1e-3:  # fotograma de blanco seguro
            for e in haces.values():
                a, b = e[0], e[1]
                if a == NEG:
                    s_ = b
                elif b == NEG:
                    s_ = a
                elif a > b:
                    s_ = a + log1p(exp(b - a))
                else:
                    s_ = b + log1p(exp(a - b))
                e[0] = s_ + pb
                e[1] = NEG
            continue
        At, It, Vt = A[t], I[t], V[t]
        cand = [(chars[It[k] - 1], Vt[k]) for k in range(K) if At[k]]
        pb_ok = pb > umbral - 3
        nuevos: dict = {}
        for pref, (b, nb, l) in haces.items():
            if b == NEG:
                tot = nb
            elif nb == NEG:
                tot = b
            elif b > nb:
                tot = b + log1p(exp(nb - b))
            else:
                tot = nb + log1p(exp(b - nb))
            e = nuevos.get(pref)
            if e is None:
                e = nuevos[pref] = [NEG, NEG, l]
            if pb_ok:
                x, y = e[0], tot + pb
                if x == NEG:
                    e[0] = y
                elif y == NEG:
                    pass
                elif x > y:
                    e[0] = x + log1p(exp(y - x))
                else:
                    e[0] = y + log1p(exp(x - y))
            ult = pref[-1] if pref else None
            dist = None
            for c, p in cand:
                if c == ult:
                    x, y = e[1], nb + p
                    if x == NEG:
                        e[1] = y
                    elif y == NEG:
                        pass
                    elif x > y:
                        e[1] = x + log1p(exp(y - x))
                    else:
                        e[1] = y + log1p(exp(x - y))
                    base = b + p
                else:
                    base = tot + p
                if base == NEG:
                    continue
                npref = pref + c
                e2 = nuevos.get(npref)
                if e2 is None:
                    if usar_lm:
                        if dist is None:
                            dist = dpref.get(pref)
                            if dist is None:
                                dist = dpref[pref] = lm.dist(contexto + pref)
                        j = lmi.get(c)
                        if j is None:
                            j = lmi[c] = ids_lm.get(c.translate(TABLA), otro)
                        lmp = l + alfa * dist[j]
                    else:
                        lmp = l
                    e2 = nuevos[npref] = [NEG, base, lmp]
                    continue
                x = e2[1]
                if x == NEG:
                    e2[1] = base
                elif x > base:
                    e2[1] = x + log1p(exp(base - x))
                else:
                    e2[1] = base + log1p(exp(x - base))
        if len(nuevos) > ancho:
            def clave(kv):
                b_, nb_, l_ = kv[1]
                if b_ == NEG:
                    s_ = nb_
                elif nb_ == NEG:
                    s_ = b_
                elif b_ > nb_:
                    s_ = b_ + log1p(exp(nb_ - b_))
                else:
                    s_ = nb_ + log1p(exp(b_ - nb_))
                return -(s_ + l_ + beta * len(kv[0]))
            haces = dict(sorted(nuevos.items(), key=clave)[:ancho])
        else:
            haces = nuevos
    final = []
    for k, (b, nb, l) in haces.items():
        fin = alfa * lm.logp(contexto + k, "\n") if usar_lm else 0.0
        final.append((k, _lae(b, nb) + l + fin + beta * len(k)))
    final.sort(key=lambda c: -c[1])
    return final[:nmejores]


def _emisiones(v, ii, etiquetas):
    """[T, len(etiquetas)] log-prob de cada etiqueta por fotograma (lo que no está entre las K mejores: la peor - 2)."""
    piso = v.min(1) - 2.0
    E = np.repeat(piso[:, None], len(etiquetas), 1)
    for j, u in enumerate(etiquetas):
        m = ii == u
        f = m.any(1)
        if f.any():
            E[f, j] = np.where(m[f], v[f], -np.inf).max(1)
    return E


def _estados(texto, chars_id):
    ids = [chars_id.get(c, -1) for c in texto]
    uni = sorted(set(ids) | {0})
    pos = {u: j for j, u in enumerate(uni)}
    est = np.zeros(2 * len(ids) + 1, np.int64)
    est[1::2] = [pos[i] for i in ids]
    salto = np.zeros(len(est), bool)  # se puede saltar el blanco anterior (etiqueta distinta de la de dos atrás)
    salto[3::2] = est[3::2] != est[1:-2:2]
    return uni, est, salto


def ctc_logp(v, ii, chars_id, texto) -> float:
    """log P(texto | lector), suma sobre alineamientos (algoritmo hacia delante)."""
    T = v.shape[0]
    if not texto:
        return float(_emisiones(v, ii, [0]).sum())
    uni, est, salto = _estados(texto, chars_id)
    E = _emisiones(v, ii, uni)[:, est]
    S = len(est)
    a = np.full(S, -np.inf)
    a[0], a[1] = E[0, 0], E[0, 1]
    for t in range(1, T):
        b = a.copy()
        b[1:] = np.logaddexp(b[1:], a[:-1])
        s2 = np.full(S, -np.inf)
        s2[2:] = a[:-2]
        b = np.where(salto, np.logaddexp(b, s2), b)
        a = b + E[t]
    return float(np.logaddexp(a[-1], a[-2]))


def alinear(v, ii, chars_id, texto):
    """Viterbi: fotograma del pico y probabilidad de cada carácter del texto."""
    T = v.shape[0]
    if not texto:
        return [], []
    uni, est, salto = _estados(texto, chars_id)
    E = _emisiones(v, ii, uni)[:, est]
    S = len(est)
    a = np.full(S, -np.inf)
    a[0], a[1] = E[0, 0], E[0, 1]
    atras = np.zeros((T, S), np.int8)
    for t in range(1, T):
        c0 = a
        c1 = np.full(S, -np.inf); c1[1:] = a[:-1]
        c2 = np.full(S, -np.inf); c2[2:] = a[:-2]
        c2 = np.where(salto, c2, -np.inf)
        m = np.stack([c0, c1, c2])
        k = m.argmax(0)
        atras[t] = k
        a = m[k, np.arange(S)] + E[t]
    s = S - 1 if a[-1] >= a[-2] else S - 2
    cam = np.zeros(T, np.int64)
    for t in range(T - 1, -1, -1):
        cam[t] = s
        s -= int(atras[t, s])
    frames, confs = [], []
    for j in range(len(texto)):
        ts = np.nonzero(cam == 2 * j + 1)[0]
        if len(ts) == 0:
            frames.append(frames[-1] if frames else 0); confs.append(0.0)
            continue
        e = E[ts, 2 * j + 1]
        k = int(e.argmax())
        frames.append(int(ts[k])); confs.append(float(min(1.0, math.exp(e[k]))))
    return frames, confs


def _base(c: str) -> str:
    return "".join(x for x in unicodedata.normalize("NFD", c) if unicodedata.category(x) != "Mn").lower()


def _signos(t: str) -> str:
    return "".join(ch for ch in t if not ch.isalnum() and not ch.isspace())


def cambio_permitido(a: str, b: str) -> bool:
    """¿Puede el modelo de lengua cambiar el trozo `a` (lo leído) por `b`? Eso lo decide la imagen, no la lengua:
    no si es solo cosa de diacríticos o de mayúsculas, no si toca cifras, y no si quita, pone o cambia signos de
    puntuación (salvo un signo leído donde había una letra: «socia!» -> «social»)."""
    if _base(a) == _base(b):
        return False
    if any(ch.isdigit() for ch in a + b):
        return False
    sa, sb = _signos(a), _signos(b)
    if sa != sb and not (sa and not sb and any(ch.isalpha() for ch in b)):
        return False
    return True


def fundir(voraz: str, cand: str, permitido=cambio_permitido, confs=None, duda_car: float = 1.01) -> tuple:
    """Aplica a `voraz` solo los cambios permitidos de `cand`. -> (texto, número de caracteres cambiados).
    Con `confs` (confianza del lector en cada carácter de `voraz`), un cambio que toca caracteres leídos solo entra si
    alguno de ellos tiene confianza < duda_car: lo que el lector leyó seguro no se toca.
    Los espacios solo se ponen o se quitan entre dos letras («ej ercicios», «redimiriá»), nunca junto a un signo."""
    sm = difflib.SequenceMatcher(None, voraz, cand, autojunk=False)
    out, n = [], 0

    def seguro(i0, i1):
        return confs is not None and i1 > i0 and len(confs) == len(voraz) and min(confs[i0:i1]) >= duda_car

    def letra(i):
        return 0 <= i < len(voraz) and voraz[i].isalpha()

    for op, i0, i1, j0, j1 in sm.get_opcodes():
        if op == "equal":
            out.append(voraz[i0:i1])
            continue
        a, b = voraz[i0:i1], cand[j0:j1]
        if op == "replace" and len(a) == len(b):  # carácter a carácter: se salvan los cambios de letra y se respetan las tildes
            for k, (x, y) in enumerate(zip(a, b)):
                if x != y and permitido(x, y) and not seguro(i0 + k, i0 + k + 1) and not (x == " " or y == " "):
                    out.append(y); n += 1
                else:
                    out.append(x)
        elif not a.strip() and not b.strip():  # solo espacios
            if letra(i0 - 1) and letra(i1):
                out.append(b); n += 1
            else:
                out.append(a)
        elif permitido(a, b) and not seguro(i0, i1) and a[:1] != " " and a[-1:] != " " and b[:1] != " " and b[-1:] != " ":
            out.append(b); n += max(len(a), len(b))
        else:
            out.append(a)
    return "".join(out), n


class Decodificador:
    def __init__(self, carpeta_lm: str | None, alfa=0.5, beta=1.0, ancho=8, umbral=-7.0, duda=0.9, delta=6.0,
                 max_cambios=6, max_frac=0.25, solo_letras=True, duda_car=1.01, lm=None, lmn=None, mlmn=0.0):
        self.lm = lm if lm is not None else (NGrama(carpeta_lm) if carpeta_lm else None)
        self.alfa, self.beta, self.ancho, self.umbral = alfa, beta, ancho, umbral
        self.duda, self.delta, self.max_cambios, self.max_frac, self.solo_letras = duda, delta, max_cambios, max_frac, solo_letras
        self.duda_car = duda_car
        self.lmn, self.mlmn = lmn, mlmn  # LM neuronal como puerta: el cambio debe ganar mlmn nats a la lectura voraz
        self._ids: dict = {}

    def _chars_id(self, chars):
        k = id(chars)
        r = self._ids.get(k)
        if r is None:
            r = self._ids[k] = {c: i + 1 for i, c in enumerate(chars)}
        return r

    def dudoso(self, v) -> bool:
        return bool((v.max(1) < math.log(self.duda)).any()) if len(v) else False

    def proponer(self, v, ii, chars, texto_voraz: str, contexto: str = "", confs=None, ant: str = "") -> str:
        """Texto final (voraz con los cambios aceptados)."""
        if not self.dudoso(v):
            return texto_voraz
        cand = haz(v, ii, chars, self.lm, self.alfa, self.beta, self.ancho, self.umbral, contexto=contexto)[0][0]
        if cand == texto_voraz:
            return texto_voraz
        if self.solo_letras:
            nuevo, n = fundir(texto_voraz, cand, confs=confs, duda_car=self.duda_car)
        else:
            nuevo = cand
            n = sum(max(i1 - i0, j1 - j0) for op, i0, i1, j0, j1 in difflib.SequenceMatcher(None, texto_voraz, cand, autojunk=False).get_opcodes() if op != "equal")
        if nuevo == texto_voraz or n > self.max_cambios or n > self.max_frac * max(4, len(texto_voraz)):
            return texto_voraz
        if self.delta is not None:
            ci = self._chars_id(chars)
            if ctc_logp(v, ii, ci, nuevo) < ctc_logp(v, ii, ci, texto_voraz) - self.delta:
                return texto_voraz
        if self.lmn is not None:
            sv, sn = self.lmn.puntuar([texto_voraz.translate(TABLA), nuevo.translate(TABLA)], anterior=(ant or "").translate(TABLA))
            if sn - sv < self.mlmn:
                return texto_voraz
        return nuevo

    def decodificar(self, v, ii, chars, voraz: tuple, contexto: str = "", ant: str = "") -> tuple:
        """voraz: (texto, conf, confs, fotogramas) de LectorCTC._decodificar. Devuelve la misma tupla. `ant`: texto del renglón anterior en el
        orden de lectura (lo usa la puerta del LM neuronal)."""
        v = np.asarray(v, np.float32)
        if v.shape[0] == 0:
            return voraz
        ii = np.asarray(ii)
        nuevo = self.proponer(v, ii, chars, voraz[0], contexto, confs=voraz[2], ant=ant)
        if nuevo == voraz[0]:
            return voraz
        frames, confs = alinear(v, ii, self._chars_id(chars), nuevo)
        return nuevo, voraz[1], confs, frames


# ---------------------------------------------------------------- integración en el producto
# Ajuste medido en validación de GRESEL (NOTAS-prensa-lengua.md, mitad A, control en B): alfa 0,3, beta 1, delta 8,
# duda_car 0,95; sin tocar tildes ni mayúsculas (solo_letras).
AJUSTE = dict(alfa=0.3, beta=1.0, delta=8.0, duda_car=0.95, ancho=8, umbral=-7.0, duda=0.9, max_cambios=6, max_frac=0.25)


# Ligero y Mínimo (lengua, 10-11: val completa, Ligero 1,699 -> 1,660 y Mínimo 1,862 -> 1,794 de CER de renglón; LM podado n5p, 8 MB)
AJUSTE_CLASICO = dict(alfa=0.2, beta=1.0, delta=12.0, duda_car=0.9, ancho=8, umbral=-7.0, duda=0.9, max_cambios=6, max_frac=0.25)


class LMNeuronalONNX:
    """LM neuronal de caracteres de «lengua» (GPT pequeño condicionado por el renglón anterior, `decod_prensa_lmn_epoca`) exportado a ONNX
    (maqueta/articulos_red/exportar.py: `lm_fp32.onnx` + `lm.json`, 42 MB). `puntuar(candidatas, anterior)` -> log P(candidata + fin | anterior)
    por candidata (misma cuenta que decod/lm_neuronal.py:LMNeuronal.puntuar)."""

    def __init__(self, carpeta: str, hilos: int = 2):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = hilos, 1
        self.s = ort.InferenceSession(os.path.join(carpeta, "lm_fp32.onnx"), so, providers=["CPUExecutionProvider"])
        self.c = json.load(open(os.path.join(carpeta, "lm.json")))
        self.idx = {ch: i + self.c["especiales"] for i, ch in enumerate(self.c["chars"])}

    def _cod(self, t: str) -> list:
        return [self.idx.get(ch, self.c["DESC"]) for ch in t.translate(TABLA)]

    def puntuar(self, cands: list, anterior: str = "") -> list:
        c = self.c
        a = self._cod(anterior)[-c["MAX_ANT"]:]
        seqs = []
        for t in cands:
            b = self._cod(t)[: c["MAX_ACT"]]
            seqs.append(([c["INI"]] + a + [c["SEP"]] + b + [c["FIN"]], len(a) + 1))
        T = max(len(i) for i, _ in seqs)
        X = np.full((len(seqs), T), c["PAD"], np.int64)
        M = np.zeros((len(seqs), T), np.float32)
        for k, (ids, ini) in enumerate(seqs):
            X[k, : len(ids)] = ids
            M[k, ini: len(ids) - 1] = 1.0  # pesos sobre los objetivos X[1:]: la candidata y el fin
        return [float(v) for v in self.s.run(None, {"X": X, "M": M})[0]]


def construir(carpeta_lm: str, lmn=None, mlmn: float = 0.0, **ajuste) -> Decodificador:
    """Decodificador del modo prensa. `lmn`: LM neuronal opcional (objeto con `puntuar(textos, anterior=)`, ver
    decod/lm_neuronal.py), usado como puerta: el cambio debe ganar `mlmn` nats a la lectura voraz."""
    return Decodificador(carpeta_lm, lmn=lmn, mlmn=mlmn, **{**AJUSTE, **ajuste})


class Corrector:
    """Sitio para el corrector aprendido (escalón 2 de «lengua»): un objeto con `corregir(texto, conf, confs, frames, v, ii,
    chars) -> (texto, conf, confs, frames)` que se aplica DESPUÉS del haz (LectorCTC.corrector)."""
