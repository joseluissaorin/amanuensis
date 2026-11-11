"""Modo prensa (prensa.py): orden de Breuel, cortes por el canal y renglones repetidos, con páginas sintéticas."""
from amanuensis.prensa import deduplicar, orden_breuel, partir_por_canales
from amanuensis.salidas import Linea

H = 20  # alto de renglón


def columna(x0, x1, y0, n, pref):
    return [Linea([x0, y0 + k * H, x1, y0 + k * H + H - 2], f"{pref}{k} texto de prueba") for k in range(n)]


def test_dos_columnas_bajo_un_titular_y_otra_noticia_debajo():
    titular = [Linea([0, 0, 610, 40], "TITULAR A DOS COLUMNAS")]
    izq = columna(0, 300, 50, 10, "a")  # renglones alineados con los de la derecha: el corte por filas los intercalaba
    der = columna(310, 610, 50, 10, "b")
    abajo = [Linea([0, 270, 610, 300], "OTRO TITULAR")] + columna(0, 300, 310, 3, "c") + columna(310, 610, 310, 3, "d")
    ls = abajo + der + titular + izq  # desordenadas a propósito
    o = [ls[i].texto for i in orden_breuel([l.caja for l in ls], H)]
    esperado = (["TITULAR A DOS COLUMNAS"] + [l.texto for l in izq] + [l.texto for l in der] + ["OTRO TITULAR"]
                + [l.texto for l in abajo[1:4]] + [l.texto for l in abajo[4:]])
    assert o == esperado


def test_cajas_que_invaden_la_columna_vecina():
    # la columna izquierda se mete 12 px en la derecha: no por eso son la misma columna
    izq = columna(0, 322, 0, 8, "a")
    der = columna(310, 620, 0, 8, "b")
    ls = der + izq
    o = [ls[i].texto for i in orden_breuel([l.caja for l in ls], H)]
    assert o == [l.texto for l in izq] + [l.texto for l in der]


def test_corte_por_el_canal():
    izq = columna(0, 300, 0, 12, "a")
    der = columna(320, 620, 0, 12, "b")
    # el renglón 6 sale unido: «a6 …» y «b6 …» en una caja que cruza el canal (x = 310)
    t = "uno dos tres cuatro" + " " + "cinco seis siete ocho"
    xs = [k * 300 / 19 for k in range(19)] + [310] + [320 + k * 300 / 21 for k in range(21)]
    unido = Linea([0, 120, 620, 138], t, 0.99, xs, confs=[0.99] * len(t))
    ls = [l for l in izq if l.caja[1] != 120] + [l for l in der if l.caja[1] != 120] + [unido]
    out, n = partir_por_canales(ls)
    assert n == 1
    textos = {l.texto for l in out}
    assert "uno dos tres cuatro" in textos and "cinco seis siete ocho" in textos


def test_renglon_repetido_al_partir_una_caja_doble():
    a = Linea([0, 0, 300, 20], "de su pieza sinfonica en que ex-", 0.95)
    b = Linea([0, 2, 300, 20], "de su pieza sinfonica en que ex-", 0.99)  # trozo releído de una caja doble
    c = Linea([0, 22, 300, 40], "presa su alegria por el logro de", 0.99)
    out = deduplicar([a, b, c], [False, True, True])
    assert [l.texto for l in out] == [a.texto, c.texto] and out[0] is a
