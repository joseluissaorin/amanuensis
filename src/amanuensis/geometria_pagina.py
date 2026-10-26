"""Orientación de la página por GEOMETRÍA (agente «orientacion», 19-10-2026; sin red nueva, solo numpy + opencv).

Pasos (todos a baja resolución salvo la asimetría):
  1. EXIF: `abrir()` aplica la orientación EXIF de las fotos (ImageOps.exif_transpose) antes de nada.
  2. Ángulo libre módulo 90° (`angulo_grueso`): nitidez del perfil de tinta en [-45°, 45°) sobre una miniatura con el
     papel normalizado; luego se afina con la dirección media robusta de las líneas base detectadas (`afinar`).
  3. Eje (horizontal o vertical): el detector de renglones en 0° y en 90°; gana el giro con más renglones largos, rectos y
     paralelos (`puntuar_renglones`: Σ longitud útil × paralelismo; los trozos cortos de texto girado apenas cuentan).
  4. 0° frente a 180° (`asimetria`): en la escritura latina (y griega y cirílica) hay más ascendentes que descendentes y la
     línea base es un borde plano y nítido: perfil de tinta por encima y por debajo del núcleo de cada renglón, en muchos
     renglones. Si la asimetría no es clara (u otras escrituras), decide el clasificador o la lectura (respaldo).

`orientar_geom(img, detectar)` -> (imagen derecha, giro_total_grados_antihorarios, info). `detectar(gris_uint8)` devuelve
renglones [{"caja", "base", "vertical"}] en px de la imagen dada (maqueta mínima con orientar=False o det-mini).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageOps


# ---------------------------------------------------------------- 1. EXIF
def abrir(ruta_o_img) -> Image.Image:
    """Image.open con la orientación EXIF aplicada (las fotos de móvil guardan los píxeles del sensor y un giro aparte)."""
    im = Image.open(ruta_o_img) if not isinstance(ruta_o_img, Image.Image) else ruta_o_img
    try:
        o = im.getexif().get(0x0112, 1)
    except Exception:
        o = 1
    if o in (None, 1) or getattr(im, "n_frames", 1) > 1:
        return im
    try:
        return ImageOps.exif_transpose(im)
    except Exception:
        return im


# ---------------------------------------------------------------- utilidades
def _gris(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    return np.asarray(img.convert("L"))


def negativa(g: np.ndarray) -> bool:
    """¿Página en negativo (texto claro sobre fondo oscuro, «modo oscuro», microfilm negativo)?"""
    m = g[:: max(1, g.shape[0] // 200), :: max(1, g.shape[1] // 200)]
    return float(np.median(m)) < 100


def _estirar(g: np.ndarray) -> np.ndarray:
    """Contraste estirado entre los percentiles 1 y 99 (páginas grises o en negativo muy apagadas)."""
    m = g[:: max(1, g.shape[0] // 300), :: max(1, g.shape[1] // 300)]
    lo, hi = np.percentile(m, [1, 99])
    if hi - lo < 1 or (lo < 30 and hi > 225):
        return g
    return np.clip((g.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def positiva(g: np.ndarray) -> np.ndarray:
    return 255 - g if negativa(g) else g


def tinta(g: np.ndarray, lado: int | None = None, binaria: bool = True) -> np.ndarray:
    """Tinta en [0, 1] con el papel normalizado (sombras y viñeteado de las fotos fuera): el papel es el máximo local.
    Con `binaria`, solo la tinta clara (umbral relativo al percentil 99): quita la textura del papel y el ruido."""
    import cv2

    g = positiva(g)
    if lado:
        s = lado / max(g.shape)
        if s < 1:
            g = cv2.resize(g, (max(1, round(g.shape[1] * s)), max(1, round(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    k = max(3, (max(g.shape) // 25) | 1)
    fondo = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    fondo = cv2.blur(fondo, (k, k)).astype(np.float32)
    t = np.clip((fondo - g.astype(np.float32)) / np.maximum(fondo, 30.0), 0, 1)
    if binaria:
        p99 = float(np.percentile(t[:: 2, :: 2], 99.5)) if t.size else 0.0
        u = max(0.18, 0.45 * p99)
        return (t >= u).astype(np.float32)
    t[t < 0.12] = 0
    return t


def rotar(img: Image.Image, grados: float, fondo=255) -> Image.Image:
    """Giro antihorario (convención de PIL) con el lienzo ampliado, para no perder esquinas."""
    if abs(grados) < 1e-6:
        return img
    if abs(grados % 90) < 1e-6:
        return img.rotate(grados, expand=True)
    f = fondo if img.mode == "L" else (fondo,) * len(img.getbands())
    return img.rotate(grados, resample=Image.BICUBIC, expand=True, fillcolor=f)


# ---------------------------------------------------------------- 2. ángulo libre
def _nitidez_perfiles(t: np.ndarray, angulos) -> tuple[np.ndarray, np.ndarray]:
    """Para cada ángulo, nitidez del perfil horizontal (renglones horizontales) y vertical de la tinta girada."""
    import cv2

    h, w = t.shape
    d = int(math.ceil(math.hypot(h, w))) + 2
    lienzo = np.zeros((d, d), np.float32)
    y0, x0 = (d - h) // 2, (d - w) // 2
    lienzo[y0:y0 + h, x0:x0 + w] = t
    c = (d / 2, d / 2)
    sh, sv = [], []
    for a in angulos:
        M = cv2.getRotationMatrix2D(c, float(a), 1.0)
        r = cv2.warpAffine(lienzo, M, (d, d), flags=cv2.INTER_LINEAR)
        ph, pv = r.sum(1), r.sum(0)
        sh.append(float(((ph[1:] - ph[:-1]) ** 2).sum()))
        sv.append(float(((pv[1:] - pv[:-1]) ** 2).sum()))
    return np.asarray(sh), np.asarray(sv)


def angulo_grueso(g, lado: int = 400, rango: float = 45.0) -> dict:
    """Inclinación módulo 90° en [-rango, rango) por la nitidez del perfil (giro antihorario que la endereza, convención de
    PIL). Devuelve {"angulo", "contraste" (nitidez del mejor / mediana), "h", "v"}: h y v son la nitidez de los perfiles
    horizontal y vertical en ese ángulo (pista del eje)."""
    t = tinta(_gris(g), lado)
    if t.mean() < 1e-3:
        return {"angulo": 0.0, "contraste": 1.0, "h": 0.0, "v": 0.0}
    # cv2.getRotationMatrix2D con ángulo positivo gira antihorario, como PIL
    A = np.arange(-rango, rango, 1.0)
    sh, sv = _nitidez_perfiles(t, A)
    s = np.maximum(sh, sv)  # módulo 90: un renglón vertical también es un renglón (se decide el eje después)
    k = int(np.argmax(s))
    fino = np.arange(A[k] - 1.0, A[k] + 1.0 + 1e-6, 0.2)
    fh, fv = _nitidez_perfiles(t, fino)
    fs = np.maximum(fh, fv)
    j = int(np.argmax(fs))

    def modulo(a):
        a = float(a)
        return a - 90 if a >= rango else a + 90 if a < -rango else a

    a = modulo(fino[j])
    # el pico de cada perfil por separado: los bordes rectos del escáner (la sombra del lomo, el canto de la hoja) dan
    # un perfil vertical nítido en 0° aunque los renglones vayan torcidos (Máscara, p. 63: bordes en 0°, renglones en
    # 2,4°); orientar_geom decide con el eje de los renglones
    picos = {}
    for cual, perfil in (("h", sh), ("v", sv)):
        kp = int(np.argmax(perfil))
        f2 = np.arange(A[kp] - 1.0, A[kp] + 1.0 + 1e-6, 0.2)
        p2 = _nitidez_perfiles(t, f2)[0 if cual == "h" else 1]
        jp = int(np.argmax(p2))
        picos[cual] = {"angulo": round(modulo(f2[jp]), 2), "nitidez": float(p2[jp])}
    return {"angulo": round(a, 2), "contraste": float(fs[j] / max(1e-9, np.median(s))), "h": float(fh[j]), "v": float(fv[j]),
            "picos": picos}


# ---------------------------------------------------------------- 3. puntuación de renglones
def _ajuste(base):
    b = np.asarray(base, np.float64)
    if len(b) < 2:
        return None
    d = b[-1] - b[0]
    L = float(np.sqrt(((b[1:] - b[:-1]) ** 2).sum(1)).sum())
    if L < 1:
        return None
    ang = math.degrees(math.atan2(-d[1], d[0]))  # positivo = sube hacia la derecha
    # rectitud: desviación de los puntos respecto a la cuerda, relativa a la longitud
    u = d / (np.linalg.norm(d) + 1e-9)
    n = np.array([-u[1], u[0]])
    dev = float(np.abs((b - b[0]) @ n).max()) / L
    return L, ang, dev


def puntuar_renglones(lineas, diag: float) -> dict:
    """Σ longitud útil de los renglones horizontales × paralelismo, relativa a la diagonal de la página.
    Longitud útil = longitud × min(1, longitud / (6 alturas)) × rectitud: los trozos de texto girado (cortos) casi no
    cuentan; los renglones verticales restan."""
    L_, A_, W_ = [], [], []
    vert = 0.0
    for l in lineas:
        c = l["caja"]
        h = max(1.0, c[3] - c[1])
        if l.get("vertical"):
            vert += max(0.0, c[3] - c[1])
            continue
        r = _ajuste(l.get("base") or [[c[0], c[3]], [c[2], c[3]]])
        if r is None:
            continue
        L, ang, dev = r
        util = L * min(1.0, L / (6 * h)) * math.exp(-dev / 0.05)
        L_.append(L)
        A_.append(ang)
        W_.append(util)
    if not W_:
        return {"puntos": -vert / diag, "n": 0, "angulo": 0.0, "dispersion": 90.0, "largos": 0}
    W = np.asarray(W_)
    A = np.asarray(A_)
    med = _mediana_pesada(A, W)
    disp = float(np.sqrt((W * (A - med) ** 2).sum() / W.sum()))
    par = 1.0 / (1.0 + disp / 3.0)
    return {"puntos": float((W.sum() * par - vert) / diag), "n": len(W_), "angulo": float(med), "dispersion": disp,
            "largos": int(sum(w > 0.5 * l for w, l in zip(W_, L_)))}


def _mediana_pesada(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1])])


# ---------------------------------------------------------------- 4. asimetría tipográfica (0° frente a 180°)
def asimetria(g: np.ndarray, lineas, max_lineas: int = 60) -> dict:
    """Para cada renglón horizontal largo: perfil de tinta por filas en una ventana amplia centrada en su caja; núcleo = la
    banda densa (≥ 45 % del máximo) más cercana al centro; tinta por encima (ascendentes, mayúsculas, tildes) y por debajo
    (descendentes) en 0,7 alturas de núcleo, y nitidez de los dos bordes del núcleo (la línea base es el borde plano).
    Devuelve {"a": mediana de (arriba − abajo)/(arriba + abajo) (> 0 = derecha), "b": mediana de (borde inferior − borde
    superior) normalizada (> 0 = derecha), "voto": fracción de renglones con a > 0, "n"}."""
    t = tinta(g)
    H, W = t.shape
    cand = [l for l in lineas if not l.get("vertical")]
    cand = sorted(cand, key=lambda l: -(l["caja"][2] - l["caja"][0]))[:max_lineas]
    sa, sb, pesos = [], [], []
    for l in cand:
        x0, y0, x1, y1 = l["caja"]
        h = y1 - y0
        if h < 6 or x1 - x0 < 4 * h:
            continue
        cy = 0.5 * (y0 + y1)
        ya, yb = int(max(0, cy - 1.1 * h)), int(min(H, cy + 1.1 * h))
        xa, xb = int(max(0, x0)), int(min(W, x1))
        if yb - ya < 8 or xb - xa < 8:
            continue
        p = t[ya:yb, xa:xb].mean(1).astype(np.float64)
        if p.max() < 0.02:
            continue
        k = max(1, int(round(h / 12)))
        if k > 1:
            p = np.convolve(p, np.ones(k) / k, "same")
        den = p >= 0.45 * p.max()
        # bandas densas contiguas; la más cercana al centro de la caja
        idx = np.nonzero(den)[0]
        grupos = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        c0 = cy - ya
        g_ = min(grupos, key=lambda gr: min(abs(gr[0] - c0), abs(gr[-1] - c0), 0 if gr[0] <= c0 <= gr[-1] else 1e9))
        top, bot = int(g_[0]), int(g_[-1])
        xh = bot - top + 1
        if xh < 3 or xh > 1.2 * h:
            continue
        m = max(2, int(round(0.7 * xh)))
        arriba = p[max(0, top - m):top].sum()
        abajo = p[bot + 1:bot + 1 + m].sum()
        if top - m < 0 or bot + 1 + m > len(p) or arriba + abajo < 1e-6:
            continue
        sa.append((arriba - abajo) / (arriba + abajo))
        dp = np.diff(p)
        e_sup = dp[max(0, top - 2):top + 2].max(initial=0)  # subida de tinta al entrar en el núcleo por arriba
        e_inf = -dp[max(0, bot - 2):bot + 2].min(initial=0)  # bajada al salir por abajo (línea base)
        sb.append((e_inf - e_sup) / (e_inf + e_sup + 1e-9))
        pesos.append(x1 - x0)
    if not sa:
        return {"a": 0.0, "b": 0.0, "voto": 0.5, "n": 0}
    w = np.asarray(pesos, np.float64)
    a = np.asarray(sa)
    b = np.asarray(sb)
    return {"a": _mediana_pesada(a, w), "b": _mediana_pesada(b, w), "voto": float((w * (a > 0)).sum() / w.sum()), "n": len(sa)}


# ---------------------------------------------------------------- todo junto
def girar_gris(g: np.ndarray, grados: float) -> np.ndarray:
    return _gris(rotar(Image.fromarray(g), grados))


def _miniatura(g: np.ndarray, lado: int) -> np.ndarray:
    esc = lado / max(g.shape)
    if esc >= 1:
        return g
    return np.asarray(Image.fromarray(g).resize((max(1, round(g.shape[1] * esc)), max(1, round(g.shape[0] * esc))), Image.BILINEAR))


def _escalar(ls, f):
    return [{**l, "caja": [v * f for v in l["caja"]], "base": [[x * f, y * f] for x, y in (l.get("base") or [])]} for l in ls]


# política de 0°/180° (ajustada en validación: bancos, pg_eval, fotos sintéticas y examen v2 DEV; nunca el sellado)
ASIM_UMBRAL = 0.08  # |asimetría| mínima
ASIM_VOTO = 0.6  # fracción mínima de renglones que votan en el mismo sentido
ASIM_MIN_RENGLONES = 4
CLASIF_SEGURO = 0.9


def orientar_geom_v1(img: Image.Image, detectar, lado_det: int = 640, clasificador=None, leer_puntos=None,
                  umbral_angulo: float = 0.3, margen_eje: float = 1.5, lado_asim: int = 1280, todas: bool = False) -> tuple:
    """-> (imagen derecha (PIL, mismo modo), giro total antihorario en grados (convención de PIL), info).

    detectar(g_uint8) -> renglones en px de g (se le pasa una miniatura POSITIVA de lado `lado_det`).
    clasificador(g_uint8) -> [p0, p90, p180, p270] (opcional; p. ej. la cabeza «ori» de la maqueta mínima).
    leer_puntos(img PIL) -> verosimilitud de la lectura de sus renglones más largos (respaldo para 0°/180°).
    Con `todas`, calcula todas las señales (clasificador y lectura incluidos) para medir políticas."""
    info = {}
    g0 = _gris(img)
    neg = negativa(g0)
    info["negativa"] = neg
    gpos = _estirar(255 - g0 if neg else g0)
    gp = _miniatura(gpos, lado_det)
    # la lectura de prueba, sobre la página positiva (en negativo el detector y el lector se pierden)
    img_lect = Image.fromarray(gpos) if (neg or gpos is not g0) else img
    ag = angulo_grueso(gp)
    info["grueso"] = ag
    a = ag["angulo"] if abs(ag["angulo"]) >= 1.0 else 0.0

    def eje_en(gp_, a_):
        cands = {}
        for k in (0, 90):
            gk = girar_gris(gp_, a_ + k)
            ls = detectar(gk)
            cands[k] = (gk, ls, puntuar_renglones(ls, math.hypot(*gk.shape)))
        return cands

    def mejor(c):
        return max(c[0][2]["puntos"], c[90][2]["puntos"])

    cands = eje_en(gp, a)
    if max(cands[0][2]["puntos"], cands[90][2]["puntos"]) < 0.5 and max(gpos.shape) > 1.4 * lado_det:
        gp = _miniatura(gpos, int(lado_det * 1.6))  # letra pequeña: el detector necesita más resolución
        cands = eje_en(gp, a)
        info["lado_det"] = max(gp.shape)
    if a:
        # el ángulo grueso tiene que mejorar los renglones (en la letra muy inclinada, como la Kurrent, el perfil engaña)
        c0 = eje_en(gp, 0.0)
        if mejor(cands) < 1.2 * mejor(c0):
            a, cands = 0.0, c0
            info["grueso_descartado"] = True
    s0, s90 = cands[0][2]["puntos"], cands[90][2]["puntos"]
    if max(s0, s90) < 0.05:  # sin renglones no hay pruebas: la página se deja como está
        info["eje"] = {"s0": round(s0, 3), "s90": round(s90, 3), "elegido": 0}
        info["decide"] = "sin renglones"
        info["giro_pre"] = 0.0
        return img, 0.0, info
    eje = 90 if s90 > margen_eje * max(s0, 1e-6) and s90 > 0 else 0
    dudoso = max(s0, s90) > 0 and min(s0, s90) > max(s0, s90) / margen_eje  # tablas, recibos: renglones en los dos ejes
    info["eje"] = {"s0": round(s0, 3), "s90": round(s90, 3), "elegido": eje, "dudoso": bool(dudoso)}

    def afinar(k):
        gk, ls, pr = cands[k]
        fino = pr["angulo"] if pr["largos"] >= 3 and pr["dispersion"] < 4.0 else 0.0
        corr = -fino  # girar +x (antihorario) endereza un renglón que baja hacia la derecha (ángulo −x)
        if abs(corr) >= umbral_angulo:
            gk = girar_gris(gp, a + k + corr)
            ls = detectar(gk)
        else:
            corr = 0.0
        return gk, ls, corr

    fondo = 0 if neg else 255
    if dudoso and leer_puntos is not None:
        # eje dudoso: decide la lectura de prueba en los cuatro giros (cambiar de eje o de sentido tiene que ganar por 0,15)
        lects = {}
        corrs = {}
        for k in (0, 90):
            _, _, corrs[k] = afinar(k)
            for extra in (0, 180):
                lects[(k, extra)] = leer_puntos(rotar(img_lect, a + k + corrs[k] + extra, 255))
        base = (eje, 0)
        # cambiar de eje con lecturas flojas (recibos, tablas de cifras) es arriesgado: margen 0,3; voltear, 0,15
        mejor_k = max(lects, key=lambda q: lects[q] - (0.0 if q == base else 0.15 if q[0] == eje else 0.3))
        info["lectura4"] = {f"{k}+{e}": round(v, 3) for (k, e), v in lects.items()}
        info["afinado"] = round(corrs[mejor_k[0]], 2)
        info["decide"] = "lectura (eje dudoso)"
        giro = a + mejor_k[0] + corrs[mejor_k[0]]
        info["giro_pre"] = round(giro, 2)
        giro += mejor_k[1]
        giro = ((giro + 180) % 360) - 180
        if abs(giro) < 1e-6:
            return img, 0.0, info
        im = rotar(img, giro, fondo)
        if abs(giro - 90 * round(giro / 90)) > 3.0:  # solo en giros grandes (el lienzo crece mucho)
            im = recortar_lienzo(im)
        return im, round(giro, 2), info
    gk, ls, corr = afinar(eje)
    info["afinado"] = round(corr, 2)
    giro = a + eje + corr
    # 4. 0° frente a 180°: asimetría a más resolución, con los renglones de la miniatura
    ga = _miniatura(gpos, lado_asim)
    gka = girar_gris(ga, giro)
    asim = asimetria(gka, _escalar(ls, gka.shape[0] / gk.shape[0])) if (todas or leer_puntos is None) else \
        {"a": 0.0, "b": 0.0, "voto": 0.5, "n": 0}  # con lector, la asimetría no decide (ver decidir_180)
    info["asimetria"] = asim
    po = None
    if clasificador is not None and (todas or leer_puntos is None):
        po = [float(x) for x in clasificador(gk)]
        info["clasificador"] = [round(x, 3) for x in po]
    lect = None
    if todas and leer_puntos is not None:
        lect = (leer_puntos(rotar(img_lect, giro, 255)), leer_puntos(rotar(img_lect, giro + 180, 255)))
        info["lectura"] = [round(lect[0], 3), round(lect[1], 3)]
    voltear, decide = decidir_180(asim, po, lect, lambda: (leer_puntos(rotar(img_lect, giro, 255)),
                                                         leer_puntos(rotar(img_lect, giro + 180, 255))) if leer_puntos else None)
    info["decide"] = decide
    info["giro_pre"] = round(giro, 2)
    if voltear:
        giro += 180
    giro = ((giro + 180) % 360) - 180  # en (-180, 180]
    if abs(giro) < 1e-6:
        return img, 0.0, info
    im = rotar(img, giro, 0 if neg else 255)
    if abs(giro - 90 * round(giro / 90)) > 3.0:  # solo en giros grandes (el lienzo crece mucho)
        im = recortar_lienzo(im)
    return im, round(giro, 2), info


# ---------------------------------------------------------------- versión PRUDENTE (19-10-2026, tras la medida final)
# La v1 giraba por error ~10 % de las páginas de otras escrituras y fotos en el examen: la lectura de prueba con el lector
# latino no distingue sentidos en árabe, han, devanagari… y la geometría de renglones toma las columnas verticales por
# renglones. Regla nueva (elegida en DEV: examen v2 dev, examen_dev_todo, pg_eval sin kurrent19 y bancos girados): los
# giros de 90/180/270° solo si el lector LEE (confianza media ≥ CONF_MIN) en el giro propuesto, lee claramente mejor que en
# la página tal cual (+DELTA_CONF) y la verosimilitud concuerda; los de 90/270 además solo si la geometría de renglones ve
# el eje girado sin dudas. Ante la duda, 0° (solo el ángulo libre, validado por los renglones).
CONF_MIN = 0.75  # elegidos en DEV (orientacion/politica_prudente.py): 0 giros de más en 323 páginas derechas; en ellas
DELTA_CONF = 0.1  # ninguna lectura girada supera a la derecha (la mejor alternativa queda 0,17 por debajo)
MARGEN_V = -1e9  # la verosimilitud (palabras vacías latinas) NO decide: engaña en otras escrituras; basta la confianza
CONF_SEGURA = 0.9  # con esta confianza en la página tal cual no se prueba ningún giro
MEJORA_GRUESO = 1.5  # el ángulo grueso tiene que mejorar los renglones al menos esto
PICO_RENGLONES = 1.25  # el pico del perfil de los renglones manda si es esto más nítido que en el ángulo de los bordes
INCLINACION_SIN_LIENZO = 5.0  # hasta aquí se endereza sin ampliar el lienzo
PICO_MIN_GRADOS = 2.0  # por debajo, det-mini lee bien la página torcida y enderezar una hoja combada junta renglones (Máscara, 4.ª página del PDF: 1,6°)


def _lectura(x):
    """leer_puntos puede devolver un número (verosimilitud, v1) o {"v", "conf"}."""
    if isinstance(x, dict):
        return float(x.get("v", 0.0)), float(x.get("conf", 0.0))
    return float(x), None


# segunda lectura de prueba con los lectores del modelo grande (l3 y multilingüe), para otras escrituras y para cuando
# Ligero no da señal: confianza media POR CARÁCTER (la de renglón de estos lectores sale alta incluso leyendo basura)
MC_MIN2 = 0.6
DELTA_MC2 = 0.15


def orientar_geom(img: Image.Image, detectar, lado_det: int = 640, clasificador=None, leer_puntos=None,
                  umbral_angulo: float = 0.3, margen_eje: float = 1.5, lado_asim: int = 1280, todas: bool = False,
                  leer_puntos2=None) -> tuple:
    """Versión PRUDENTE. -> (imagen derecha, giro total antihorario en grados, info).
    leer_puntos(img PIL) -> {"v": verosimilitud, "conf": confianza media} de los renglones más largos (o un número).
    Sin lector solo se endereza el ángulo libre (nunca se gira 90/180/270 a ciegas). `clasificador`, `lado_asim` y
    `todas` se aceptan por compatibilidad y no se usan."""
    info = {"version": "prudente"}
    g0 = _gris(img)
    neg = negativa(g0)
    info["negativa"] = neg
    gpos = _estirar(255 - g0 if neg else g0)
    img_lect = Image.fromarray(gpos) if (neg or gpos is not g0) else img
    fondo = 0 if neg else 255
    gp = _miniatura(gpos, lado_det)
    ag = angulo_grueso(gp)
    info["grueso"] = ag
    a = ag["angulo"] if abs(ag["angulo"]) >= 1.0 else 0.0

    def eje_en(gp_, a_):
        cands = {}
        for k in (0, 90):
            gk = girar_gris(gp_, a_ + k)
            ls = detectar(gk)
            cands[k] = (gk, ls, puntuar_renglones(ls, math.hypot(*gk.shape)))
        return cands

    def mejor(c):
        return max(c[0][2]["puntos"], c[90][2]["puntos"])

    cands = eje_en(gp, a)
    if mejor(cands) < 0.5 and max(gpos.shape) > 1.4 * lado_det:
        gp = _miniatura(gpos, int(lado_det * 1.6))
        cands = eje_en(gp, a)
    if a:
        c0 = eje_en(gp, 0.0)
        if mejor(cands) < MEJORA_GRUESO * mejor(c0):
            a, cands = 0.0, c0
            info["grueso_descartado"] = True
    s0, s90 = cands[0][2]["puntos"], cands[90][2]["puntos"]
    picos = ag.get("picos")
    if picos and not info.get("grueso_descartado") and max(s0, s90) >= 0.05:
        # el ángulo lo ganó el perfil que no es el de los renglones (bordes rectos del escáner): manda el pico del perfil de
        # los renglones si es claramente más nítido que ese perfil en el ángulo elegido
        cual = "v" if s90 > margen_eje * max(s0, 1e-6) and s90 > 0 else "h"
        p = picos[cual]
        if abs(p["angulo"] - a) >= PICO_MIN_GRADOS and p["nitidez"] >= PICO_RENGLONES * ag[cual]:
            a = p["angulo"]
            cands = eje_en(gp, a)
            s0, s90 = cands[0][2]["puntos"], cands[90][2]["puntos"]
            info["grueso_renglones"] = a
    info["eje"] = {"s0": round(s0, 3), "s90": round(s90, 3)}
    if max(s0, s90) < 0.05:
        info["decide"] = "sin renglones"
        return img, 0.0, info
    eje90 = s90 > margen_eje * max(s0, 1e-6) and s90 > 0  # la geometría ve el eje girado sin dudas
    info["eje"]["girado"] = bool(eje90)

    def afinar(k):
        gk, ls, pr = cands[k]
        fino = pr["angulo"] if pr["largos"] >= 3 and pr["dispersion"] < 4.0 else 0.0
        return -fino if abs(fino) >= umbral_angulo else 0.0

    corr0 = afinar(0)
    base = a + corr0  # la página tal cual, solo enderezada
    giro = base
    if leer_puntos is not None:
        v0, c0_ = _lectura(leer_puntos(rotar(img_lect, base, 255)))
        info["lectura"] = {"0": [round(v0, 3), None if c0_ is None else round(c0_, 3)]}
        if c0_ is None or c0_ < CONF_SEGURA or eje90:
            props = [base + 180]
            if eje90:
                c90 = afinar(90)
                props += [a + 90 + c90, a + 270 + c90]
            mejor_p = None
            for g in props:
                v, c = _lectura(leer_puntos(rotar(img_lect, g, 255)))
                info["lectura"][str(round((g - base) % 360))] = [round(v, 3), None if c is None else round(c, 3)]
                if c is None or c0_ is None:  # lector sin confianza: no se gira (prudencia)
                    continue
                if c >= CONF_MIN and c - c0_ >= DELTA_CONF and v >= v0 + MARGEN_V and (mejor_p is None or c > mejor_p[1]):
                    mejor_p = (g, c)
            if mejor_p is not None:
                giro = mejor_p[0]
                info["decide"] = "lectura"
    if leer_puntos2 is not None and giro == base:
        c0l = (info.get("lectura") or {}).get("0", [None, None])[1]
        if c0l is None or c0l < CONF_SEGURA:
            # Ligero no da señal clara (otras escrituras, manuscritos difíciles): lectores del modelo grande
            m0 = float(leer_puntos2(rotar(img_lect, base, 255)))
            props = [base + 180]
            if eje90:
                c90 = afinar(90)
                props += [a + 90 + c90, a + 270 + c90]
            info["lectura2"] = {"0": round(m0, 3)}
            mejor_p = None
            for g in props:
                mc = float(leer_puntos2(rotar(img_lect, g, 255)))
                info["lectura2"][str(round((g - base) % 360))] = round(mc, 3)
                if mc >= MC_MIN2 and mc - m0 >= DELTA_MC2 and (mejor_p is None or mc > mejor_p[1]):
                    mejor_p = (g, mc)
            if mejor_p is not None:
                giro = mejor_p[0]
                info["decide"] = "lectura grande"
    info.setdefault("decide", "derecha")
    info["afinado"] = round(corr0, 2)
    info["giro_pre"] = round(base, 2)
    giro = ((giro + 180) % 360) - 180
    if abs(giro) < 1e-6:
        return img, 0.0, info
    if abs(giro) <= INCLINACION_SIN_LIENZO:
        # una inclinación pequeña, sin ampliar el lienzo (como orientacion.enderezar): las esquinas que se pierden son
        # margen, y con el lienzo ampliado la letra queda más pequeña para det-mini (Máscara, p. 63: 2,4°)
        f = fondo if img.mode == "L" else (fondo,) * len(img.getbands())
        return img.rotate(giro, resample=Image.BICUBIC, expand=False, fillcolor=f), round(giro, 2), info
    im = rotar(img, giro, fondo)
    if abs(giro - 90 * round(giro / 90)) > 3.0:
        im = recortar_lienzo(im)
    return im, round(giro, 2), info


def recortar_lienzo(im: Image.Image, margen: float = 0.03) -> Image.Image:
    """Tras un giro libre el lienzo crece (esquinas en blanco) y la letra queda más pequeña para los detectores que escalan
    la página a un ancho fijo (det-mini): se recorta a la caja de la tinta con un margen."""
    g = np.asarray(im.convert("L"))
    t = tinta(g, 800)
    ys, xs = np.nonzero(t)
    if len(xs) < 20:
        return im
    e = g.shape[1] / t.shape[1]
    x0, x1 = np.percentile(xs, [0.02, 99.98]) * e
    y0, y1 = np.percentile(ys, [0.02, 99.98]) * e
    m = margen * max(g.shape)
    caja = (int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(g.shape[1], x1 + m)), int(min(g.shape[0], y1 + m)))
    if (caja[2] - caja[0]) * (caja[3] - caja[1]) > 0.9 * g.shape[0] * g.shape[1]:
        return im
    return im.crop(caja)


def decidir_180(asim, po, lect=None, leer=None):
    """-> (¿voltear?, quién decide). Medido en validación (NOTAS-maqueta.md, «Orientación por geometría»): con lector, la
    lectura de prueba de los renglones más largos en los dos sentidos (voltear solo si gana por 0,15) es la mejor señal en
    todos los bancos; la asimetría tipográfica acierta en impreso latino limpio, pero las fotos (curvatura, luz) la engañan
    y otras escrituras (devanagari, hebreo, tibetano) la tienen al revés; el clasificador de la maqueta mínima falla en
    dobles páginas y fotos. Sin lector: el clasificador si está seguro y, si no, la asimetría."""
    if lect is None and leer is not None:
        lect = leer()
    if lect is not None:
        return lect[1] > lect[0] + 0.15, "lectura"
    if po is not None and max(po[0], po[2]) >= CLASIF_SEGURO:
        return po[2] > po[0], "clasificador"
    n_ok = asim["n"] >= ASIM_MIN_RENGLONES
    if n_ok and asim["a"] <= -ASIM_UMBRAL and asim["voto"] <= 1 - ASIM_VOTO:
        return True, "asimetria"
    return False, "asimetria" if n_ok and asim["a"] >= ASIM_UMBRAL else "nada"


# ---------------------------------------------------------------- 5. zonas con texto girado (lomos, márgenes, cabeceras de tabla)
def renglones_girados(g: np.ndarray, detectar, cajas0, lado: int = 1024, min_largo: float = 5.0):
    """Renglones que solo se ven con la página girada 90° o 270° (texto en vertical sobre una página ya derecha).
    g: página derecha (uint8); cajas0: cajas de los renglones horizontales ya encontrados (px de g).
    -> [{"giro": 90|270, "caja": caja en px de g, "caja_girada": caja en px de la página girada a resolución completa}]."""
    H, W = g.shape
    esc = min(1.0, lado / max(H, W))
    gp = _miniatura(positiva(g), lado)
    out = []
    for k in (90, 270):
        gk = np.ascontiguousarray(np.rot90(gp, k // 90))  # np.rot90 gira antihorario, como PIL
        hk, wk = gk.shape
        for l in detectar(gk):
            if l.get("vertical"):
                continue
            x0, y0, x1, y1 = l["caja"]
            h = max(1.0, y1 - y0)
            if x1 - x0 < min_largo * h:
                continue
            # de la página girada k (antihorario) a la derecha: k=90: (x, y) -> (W' - 1 - y, x) con W' = ancho de gp
            if k == 90:
                X0, X1 = gp.shape[1] - y1, gp.shape[1] - y0
                Y0, Y1 = x0, x1
            else:
                X0, X1 = y0, y1
                Y0, Y1 = gp.shape[0] - x1, gp.shape[0] - x0
            caja = [X0 / esc, Y0 / esc, X1 / esc, Y1 / esc]
            area = max(1.0, (caja[2] - caja[0]) * (caja[3] - caja[1]))
            tapado = sum(max(0, min(caja[2], b[2]) - max(caja[0], b[0])) * max(0, min(caja[3], b[3]) - max(caja[1], b[1])) for b in cajas0)
            if tapado > 0.15 * area:
                continue
            out.append({"giro": k, "caja": caja, "caja_girada": [x0 / esc, y0 / esc, x1 / esc, y1 / esc], "alto": h / esc})
    # el mismo renglón sale en 90° (derecho) y en 270° (del revés): se agrupan por solape y se leen los dos
    grupos = []
    for r in out:
        for gr in grupos:
            a, b = gr[0]["caja"], r["caja"]
            i = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if i > 0.5 * min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])):
                gr.append(r)
                break
        else:
            grupos.append([r])
    return grupos
