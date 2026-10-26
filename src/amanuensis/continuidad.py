"""Continuidad del texto entre renglones (agente «orientacion», 19-10-2026): un modelo de lengua de caracteres diminuto
(n-gramas hasta 5, tablas con dispersión por función resumen y log-probabilidades cuantizadas a un byte, ~5 MB) que
puntúa «el renglón B sigue al renglón A»: guion de partición que recompone la palabra, frase abierta, minúscula inicial.
Sirve para elegir entre varios órdenes de lectura candidatos (columnas, dobles páginas, saltos de bloque) en Ligero y en
el modelo grande. Solo numpy.

    lm = ModeloContinuidad.cargar("continuidad.npz")
    lm.transicion("los ejem-", "plos que siguen")      # PMI en nats: > 0 = B sigue a A mejor que a cualquier cosa
    lm.puntuar_orden(textos, orden)                    # Σ de transiciones
"""
from __future__ import annotations

import math
import os
import unicodedata

import numpy as np

GUIONES = "-‐‑¬⸗="
_P = np.uint64(1099511628211)  # FNV
_M = np.uint64(0x9E3779B97F4A7C15)
NMAX = 5
LOGP_MIN = -12.0  # nats


def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    return " ".join(t.split())


def _hashes(cp: np.ndarray, n: int) -> np.ndarray:
    """Función resumen de los n-gramas que empiezan en cada posición (uint64, aritmética con desbordamiento)."""
    L = len(cp) - n + 1
    if L <= 0:
        return np.zeros(0, np.uint64)
    h = np.full(L, np.uint64(14695981039346656037), np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = (h ^ cp[k:k + L].astype(np.uint64)) * _P
    return h


def _cubo(h: np.ndarray, bits: int, sal: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        return ((h ^ np.uint64(sal * 0x5BD1E995)) * _M) >> np.uint64(64 - bits)


_MASK = (1 << 64) - 1


def _cubo_py(s: str, n: int, bits: int) -> int:
    """= _cubo(_hashes(s, n)) para una sola cadena de n caracteres, en Python puro (rápido para consultas sueltas)."""
    h = 14695981039346656037
    for ch in s:
        h = ((h ^ ord(ch)) * 1099511628211) & _MASK
    return (((h ^ (n * 0x5BD1E995)) * 0x9E3779B97F4A7C15) & _MASK) >> (64 - bits)


class ModeloContinuidad:
    """Witten-Bell interpolado: P(c|h) = λ(h) P_ML(c|h) + (1 − λ(h)) P(c|h'), con λ(h) = N(h) / (N(h) + T(h)).
    Tablas por orden n: «t» (log P_ML del n-grama, un byte; 0 = sin ver) y «l» (λ del contexto de n − 1 caracteres)."""

    def __init__(self, tablas: list[np.ndarray], lambdas: list[np.ndarray], bits: int, uniforme: float):
        self.t = tablas
        self.l = lambdas
        self.bits = bits
        self.uniforme = uniforme  # probabilidad base (1 / tamaño del alfabeto visto)
        self._cache: dict = {}

    @classmethod
    def cargar(cls, ruta: str):
        z = np.load(ruta)
        n = int(z["nmax"])
        return cls([z[f"t{k}"] for k in range(1, n + 1)], [z[f"l{k}"] for k in range(1, n + 1)], int(z["bits"]), float(z["uniforme"]))

    def guardar(self, ruta: str):
        np.savez_compressed(ruta, nmax=len(self.t), bits=self.bits, uniforme=self.uniforme,
                            **{f"t{k + 1}": t for k, t in enumerate(self.t)}, **{f"l{k + 1}": t for k, t in enumerate(self.l)})

    # ---- probabilidad
    def logp(self, ctx: str, c: str) -> float:
        """log P(c | ctx) en nats."""
        ctx = ctx[-(NMAX - 1):]
        clave = (ctx, c)
        r = self._cache.get(clave)
        if r is not None:
            return r
        p = self.uniforme
        for n in range(1, len(ctx) + 2):
            h = ctx[len(ctx) - (n - 1):] if n > 1 else ""
            lam = self.l[n - 1][_cubo_py(h, n - 1, self.bits)] / 255.0 if n > 1 else float(self.l[0][0]) / 255.0
            if lam <= 0:
                break  # contexto sin ver: los órdenes altos tampoco existen
            v = int(self.t[n - 1][_cubo_py(h + c, n, self.bits)])
            pml = math.exp(LOGP_MIN * (255 - v) / 254) if v else 0.0
            p = lam * pml + (1 - lam) * p
        r = math.log(max(p, 1e-12))
        if len(self._cache) < 500_000:
            self._cache[clave] = r
        return r

    def logp_cadena(self, ctx: str, s: str) -> float:
        tot = 0.0
        for i, c in enumerate(s):
            tot += self.logp(ctx + s[:i], c)
        return tot

    # ---- continuidad
    def transicion(self, a: str, b: str, k: int = 8) -> float:
        """PMI (nats) de los primeros k caracteres de B tras A frente a tras un espacio (comienzo de palabra cualquiera).
        Con guion de partición al final de A (letra antes y letra después), se pega la palabra."""
        a, b = normalizar(a), normalizar(b)
        if not a or not b:
            return 0.0
        b = b[:k]
        if len(a) > 1 and a[-1] in GUIONES and a[-2].isalpha() and b[0].isalpha():
            ctx = a[:-1][-(NMAX - 1):]
        else:
            ctx = (a + " ")[-(NMAX - 1):]
        return self.logp_cadena(ctx, b) - self.logp_cadena(" ", b)

    def matriz(self, textos: list[str], k: int = 8) -> np.ndarray:
        n = len(textos)
        M = np.zeros((n, n), np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    M[i, j] = self.transicion(textos[i], textos[j], k)
        return M

    def puntuar_orden(self, textos: list[str], orden: list[int], k: int = 8) -> float:
        return float(sum(self.transicion(textos[orden[t]], textos[orden[t + 1]], k) for t in range(len(orden) - 1)))


def entrenar(textos, bits: int = 21, minimo: int = 2) -> ModeloContinuidad:
    """textos: iterable de cadenas (párrafos). Cuentas exactas por cubo (2^bits) y Witten-Bell."""
    trozos = []
    for t in textos:
        t = normalizar(t)
        if len(t) >= 3:
            trozos.append(" " + t + " ")
    s = "".join(trozos)
    cp = np.frombuffer(s.encode("utf-32-le"), np.uint32)
    N = 1 << bits
    tablas, lambdas = [], []
    b_prev = None  # cubos de los (n-1)-gramas por posición
    alfabeto = len(np.unique(cp))
    for n in range(1, NMAX + 1):
        h = _hashes(cp, n)
        b = _cubo(h, bits, n).astype(np.int64)
        c_ng = np.bincount(b, minlength=N)
        if n == 1:
            tot = float(len(cp))
            pml = c_ng[b] / tot
            lam = np.full(1, round(255 * tot / (tot + alfabeto)), np.uint8)
            lam_tab = np.zeros(N, np.uint8)
            lam_tab[0] = lam[0]
        else:
            bc = b_prev[: len(b)]
            c_ctx = np.bincount(bc, minlength=N)
            par = np.unique(b * N + bc)
            tipos = np.bincount(par % N, minlength=N)
            pml = c_ng[b] / np.maximum(c_ctx[bc], 1)
            l = c_ctx / np.maximum(c_ctx + tipos, 1)
            lam_tab = np.clip(np.round(255 * l), 0, 255).astype(np.uint8)
            lam_tab[(c_ctx > 0) & (lam_tab == 0)] = 1
        ok = c_ng[b] >= (minimo if n >= 3 else 1)
        v = np.clip(np.round(255 - 254 * np.log(np.maximum(pml, 1e-30)) / LOGP_MIN), 1, 255).astype(np.uint8)
        t = np.zeros(N, np.uint8)
        t[b[ok]] = v[ok]
        tablas.append(t)
        lambdas.append(lam_tab)
        b_prev = b
    return ModeloContinuidad(tablas, lambdas, bits, 1.0 / (alfabeto + 1))


_MODELO = [None, False]


def modelo():
    """El modelo empaquetado (continuidad.npz junto a este fichero), o None."""
    if not _MODELO[1]:
        _MODELO[1] = True
        r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "continuidad.npz")
        if os.path.exists(r):
            _MODELO[0] = ModeloContinuidad.cargar(r)
    return _MODELO[0]


def elegir_orden(textos: list[str], candidatos: dict, defecto: str, margen: float = 10.0, lm=None):
    """Entre varios órdenes candidatos (nombre -> permutación de los renglones), el de más continuidad del texto; el
    candidato `defecto` se queda salvo que otro le gane por más de `margen` nats. -> (nombre, permutación)."""
    lm = lm or modelo()
    if lm is None or len(textos) < 3:
        return defecto, candidatos[defecto]
    vistos, puntos = {}, {}
    for k, o in candidatos.items():
        t = tuple(o)
        if t not in vistos:
            vistos[t] = lm.puntuar_orden(textos, o)
        puntos[k] = vistos[t]
    k = max(puntos, key=puntos.get)
    if puntos[k] - puntos[defecto] <= margen:
        k = defecto
    return k, candidatos[k]


def orden_por_columnas(cajas) -> list[int]:
    """Bloques por columna (renglones que se solapan en x y siguen en y) ordenados de izquierda a derecha y de arriba abajo
    por su esquina; dentro de cada bloque, de arriba abajo."""
    cols = []
    for i in sorted(range(len(cajas)), key=lambda i: cajas[i][1]):
        c = cajas[i]
        h = max(1.0, c[3] - c[1])
        for col in cols:
            ult = cajas[col[2][-1]]
            solape = min(c[2], col[1]) - max(c[0], col[0])
            if solape > 0.5 * min(c[2] - c[0], col[1] - col[0]) and c[1] - ult[3] < 2.5 * h:
                col[2].append(i)
                col[0], col[1] = min(col[0], c[0]), max(col[1], c[2])
                break
        else:
            cols.append([c[0], c[2], [i]])
    cols.sort(key=lambda col: (col[0], cajas[col[2][0]][1]))
    return [i for col in cols for i in col[2]]
