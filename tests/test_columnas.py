"""Columnas de prensa: comienzos alineados, trozos de renglón, cajas que cruzan el canal y el orden por la palabra
partida que une dos bloques."""

import numpy as np
from PIL import Image

from amanuensis import columnas
from amanuensis.salidas import Linea, Pagina


def test_las_sangrias_no_son_columnas():
    h = 30
    cajas = [[100, 40 * k, 900, 40 * k + h] for k in range(20)] + [[160, 40 * k, 900, 40 * k + h] for k in range(20, 26)]
    assert columnas.comienzos(cajas, h) == [100.0]
    cajas += [[1000, 40 * k, 1800, 40 * k + h] for k in range(20)]
    assert columnas.comienzos(cajas, h) == [100.0, 1000.0]


def test_los_trozos_de_un_renglon_se_unen_y_el_canal_no():
    h = 30
    g = np.full((200, 2000), 255, np.uint8)
    g[10:35, 100:900] = 0      # renglón de la columna 1 con un hueco CON tinta entre trozos
    g[10:35, 1000:1800] = 0    # el renglón vecino de la columna 2, tras un canal blanco
    cajas = [[100, 10, 400, 35], [470, 10, 900, 35], [1000, 10, 1800, 35]]
    out = columnas.unir_trozos(cajas, [100.0, 1000.0], h, g)
    assert sorted(c[0] for c in out) == [100, 1000] and any(c[2] == 900 for c in out)


def test_la_caja_que_cruza_el_canal_se_parte():
    g = np.full((400, 2000), 255, np.uint8)
    for k in range(6):
        y = 20 + 50 * k
        g[y:y + 30, 100:900] = 0
        g[y:y + 30, 960:1800] = 0
    img = Image.fromarray(g)
    cajas = [[100, 20 + 50 * k, 900, 50 + 50 * k] for k in range(5)] + [[960, 20 + 50 * k, 1800, 50 + 50 * k] for k in range(5)]
    cajas.append([100, 270, 1800, 300])  # el detector juntó los dos renglones de abajo
    out, n = columnas.partir_por_canales(img, cajas)
    assert n == 1 and any(c[0] == 100 and c[2] < 960 and c[1] == 270 for c in out)


def test_la_palabra_partida_dice_que_bloque_sigue():
    def L(x, y, t):
        return Linea([x, y, x + 400, y + 30], t, 0.99)
    ls = [L(100, 40 * k, f"izquierda renglón {k} del texto corrido") for k in range(4)]
    ls += [L(1100, 40 * k, f"derecha {k} sigue el texto") for k in range(4)]   # (debería ir al final)
    ls[4].texto = "te). En la posición estructurada"
    ls += [L(600, 400 + 40 * k, f"centro {k} con más texto") for k in range(3)] + [L(600, 520, "y estar listo para el comba-")]
    p = Pagina(2000, 1000, ls)
    frec = {"combate": 145161, "combatan": 175}.get
    assert columnas.por_continuidad(p, lambda w: frec(w, 0)) == 1
    t = [l.texto for l in p.lineas]
    assert t.index("y estar listo para el comba-") + 1 == t.index("te). En la posición estructurada")


def test_regiones_nms_y_corte_xy():
    from amanuensis import maqueta_yolo as Y
    cajas = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    assert sorted(Y._nms(cajas, np.array([0.9, 0.8, 0.7]), 0.5)) == [0, 2]
    # dos columnas y un título que las cruza: título, columna izquierda, columna derecha
    regs = [[0, 0, 1000, 80], [0, 100, 480, 900], [520, 100, 1000, 900]]
    assert Y._corte_xy(regs, [2, 1, 0]) == [0, 1, 2]


def test_la_palabra_partida_no_arranca_un_trozo_de_renglon_ni_mueve_por_una_palabra_rara():
    """Máscara, p. 63, con 0,2° de inclinación residual: «estabiliz» | «ción, inmovilidad» es un renglón partido en dos
    trozos, no el arranque de otra columna, y el «ac-» del final de la página sigue en la siguiente."""
    def L(x, y, t, w=400):
        return Linea([x, y, x + w, y + 30], t, 0.99)

    def pagina(trozo):
        ls = [L(100, 40 * k, f"izquierda renglón {k} del texto corrido") for k in range(8)]
        ls += [L(1100, 40 * k, f"derecha {k} sigue el texto") for k in range(4)]
        ls += [L(1100, 160, "estabiliz", 120), L(1300, 160, trozo, 200)]
        ls += [L(1100, 200 + 40 * k, f"derecha {k} sigue más abajo") for k in range(3)]
        ls += [L(600, 500 + 40 * k, f"centro {k} con más texto") for k in range(2)] + [L(600, 580, "le va a decir al ac-")]
        return Pagina(2000, 1000, ls)
    p = pagina("ción, inmovilidad, y")
    antes = [l.texto for l in p.lineas]
    assert columnas.por_continuidad(p, lambda w: {"acción": 90000}.get(w, 0)) == 0
    assert [l.texto for l in p.lineas] == antes
    # arriba de una columna sí, pero la única candidata es rara («ac-» + «te», 158): tampoco se mueve
    p = pagina("y otra cosa más")
    p.lineas[8].texto = "te una vez más"
    assert columnas.por_continuidad(p, lambda w: {"acte": 158}.get(w, 0)) == 0


def test_doclayout_yolo_solo_si_se_pide(monkeypatch, tmp_path):
    from amanuensis import maqueta_yolo

    (tmp_path / "modelo.onnx").write_bytes(b"")
    monkeypatch.setenv("AMANUENSIS_CACHE", str(tmp_path.parent))
    monkeypatch.delenv("AMANUENSIS_MAQUETA_YOLO", raising=False)
    monkeypatch.delenv("AMANUENSIS_MAQUETA_YOLO_SI", raising=False)
    monkeypatch.delenv("AMANUENSIS_MAQUETA_YOLO_NO", raising=False)
    assert maqueta_yolo.activa() is None  # instalado no basta
    monkeypatch.setenv("AMANUENSIS_MAQUETA_YOLO", str(tmp_path))
    assert maqueta_yolo.activa() == str(tmp_path)
    monkeypatch.setenv("AMANUENSIS_MAQUETA_YOLO_NO", "1")
    assert maqueta_yolo.activa() is None
