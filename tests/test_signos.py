"""Signos: comillas tímidas y por parejas, rayas medidas en la imagen, ⊢ y la notación del documento."""

import numpy as np

from amanuensis import signos
from amanuensis.lector import _signos_timidos, voraz
from amanuensis.salidas import Linea, Pagina

CHARS = list("abcdefghijklmnopqrstuvwxyzáS ‘’'“”\"«»")


def lp_de(secuencia, extra=None):
    """Log-probabilidades [T, C+1] con un símbolo seguro por fotograma (0 = blanco) y probabilidades extra."""
    T = len(secuencia)
    p = np.full((T, len(CHARS) + 1), 1e-6)
    for t, c in enumerate(secuencia):
        p[t, 0 if c is None else CHARS.index(c) + 1] = 1.0
    for (t, c), v in (extra or {}).items():
        p[t, CHARS.index(c) + 1] = v
        p[t, 0] = 1 - v
    p /= p.sum(1, keepdims=True)
    return np.log(p)


def test_la_comilla_timida_se_escribe_y_la_ausente_no():
    sec = [None, "S", None, "e", None, "r", None, "’", None]
    lp = lp_de(sec, {(0, "‘"): 0.3})
    assert voraz(lp, CHARS, signos=0.08).texto == "‘Ser’"
    assert voraz(lp_de(sec), CHARS, signos=0.08).texto == "Ser’"


def test_no_se_duplica_junto_a_otra_comilla():
    sec = [None, "a", None, "’", None, None]
    lp = lp_de(sec, {(4, "’"): 0.2})
    assert voraz(lp, CHARS, signos=0.08).texto == "a’"


def test_la_comilla_que_falta_de_una_pareja():
    texto = "la Ser simpática’"
    sec = []
    fot = []
    for ch in texto:
        sec += [None, ch]
        fot.append(len(sec) - 1)
    lp = lp_de(sec + [None], {(6, "‘"): 0.03})  # algo de comilla en el hueco antes de «Ser» (fotograma 6)
    l = Linea([0, 0, 100, 20], texto, 0.99, xs=list(range(len(texto))), confs=[0.99] * len(texto))
    l._ctc = (lp, fot, CHARS)
    signos.comillas_pareadas([Pagina(100, 100, [l])])
    assert l.texto == "la ‘Ser simpática’"


def _raya(g, x0, x1, y, palo=False):
    g[y:y + 3, x0:x1] = 0
    if palo:
        g[y - 12:y + 12, x0:x0 + 3] = 0


def test_rayas_por_su_medida_y_el_entrana():
    g = np.full((200, 1000), 255, np.uint8)
    lineas = []
    for k, (ancho, palo) in enumerate([(12, False), (13, False), (30, False), (31, False), (20, True)]):
        y0 = 20 + 35 * k
        _raya(g, 200, 200 + ancho, y0 + 15, palo)
        t = "ab — cd"
        l = Linea([100, y0, 400, y0 + 30], t, 0.99, xs=[110, 130, 150, 200 + ancho / 2, 250, 270, 290], confs=[0.99] * len(t))
        signos.medir_rayas(g, l)
        lineas.append(l)
    signos.decidir_rayas([Pagina(1000, 200, lineas)])
    assert [l.texto[3] for l in lineas] == ["–", "–", "—", "—", "⊢"]


def test_un_solo_grupo_de_rayas_respeta_lo_leido():
    g = np.full((120, 1000), 255, np.uint8)
    lineas = []
    for k in range(3):
        y0 = 20 + 30 * k
        _raya(g, 200, 214, y0 + 15)
        l = Linea([100, y0, 400, y0 + 30], "ab — cd", 0.99, xs=[110, 130, 150, 207, 250, 270, 290], confs=[0.99] * 7)
        signos.medir_rayas(g, l)
        lineas.append(l)
    signos.decidir_rayas([Pagina(1000, 120, lineas)])
    assert all(l.texto == "ab — cd" for l in lineas)


def test_la_notacion_del_documento():
    ls = [Linea([0, 20 * k, 900, 20 * k + 18], f"Ana dijo algo +> ‘Le gusta {k}’", 0.99) for k in range(5)]
    ls += [Linea([0, 200, 900, 218], "(+‘No todos los estudiantes’), y la escala <TODOS, ALGUNOS> se queda", 0.99),
           Linea([0, 220, 900, 238], "a >> ‘b’ y c >> ‘d’ y e >> ‘f’ y g >> ‘h’ y i >> ‘j’ y la basura > ‘Beatriz’", 0.99)]
    signos.notacion([Pagina(1000, 300, ls)])
    assert ls[5].texto.startswith("(+> ‘No todos") and "<TODOS, ALGUNOS>" in ls[5].texto
    assert ls[6].texto.endswith("basura >> ‘Beatriz’")
