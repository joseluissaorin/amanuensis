import numpy as np

from amanuensis import completo, haz_lm


def _vi(texto, chars):
    ids = [chars.index(c) + 1 for c in texto]
    T = len(ids) * 2
    v = np.full((T, 2), -9.0, np.float32)
    ii = np.zeros((T, 2), np.int64)
    for k, i in enumerate(ids):
        v[2 * k, 0], ii[2 * k, 0] = -0.01, i
        v[2 * k + 1, 0], ii[2 * k + 1, 0] = -0.01, 0
        ii[2 * k, 1], ii[2 * k + 1, 1] = 0, i
    return v, ii


def test_haz_sin_modelo_no_cambia_lo_seguro():
    chars = list("abcdefghijklmnopqrstuvwxyz ")
    v, ii = _vi("hola mundo", chars)
    d = haz_lm.Decodificador(None, **haz_lm.AJUSTE)
    voraz = ("hola mundo", 0.99, [0.99] * 10, list(range(0, 20, 2)))
    assert d.decodificar(v, ii, chars, voraz) == voraz  # ningún fotograma dudoso: sale como lo leyó el lector


def test_ajuste_medido():
    assert (haz_lm.AJUSTE["alfa"], haz_lm.AJUSTE["beta"], haz_lm.AJUSTE["delta"], haz_lm.AJUSTE["duda_car"]) == (0.3, 1.0, 8.0, 0.95)


def test_gancho_en_el_lector():
    class Falso(completo.LectorCTC):
        def __init__(self):
            self.chars = list("ab")
            self.haz = None

    class Haz:
        def decodificar(self, v, ii, chars, r):
            return ("X",) + tuple(r[1:])

    l = Falso()
    v = np.array([[-0.01, -9.0]], np.float32)
    ii = np.array([[1, 0]], np.int64)
    assert l._decodificar(v, ii, 0.0)[0] == "a"
    l.haz = Haz()
    assert l._decodificar(v, ii, 0.0)[0] == "X"


def test_puerta_neuronal_difiere_con_el_renglon_anterior_y_hay_paridad():
    import os

    import pytest

    ruta = os.environ.get("AMANUENSIS_RUTA_LM_NEURONAL") or os.path.expanduser("~/.cache/amanuensis/amanuensis-lm-neuronal-1")
    if not os.path.exists(os.path.join(ruta, "lm_fp32.onnx")):
        pytest.skip("sin el LM neuronal en la caché")
    lm = haz_lm.LMNeuronalONNX(ruta)
    # cifras de decod/lm_neuronal.py (PyTorch) con las mismas entradas: -29,0705 y -30,2501
    assert abs(lm.puntuar(["hola mundo"])[0] + 29.0705) < 0.01
    assert abs(lm.puntuar(["a los delegados del congreso de la ciudad"], anterior="el presidente de la republica recibio ayer")[0] + 30.2501) < 0.01


def test_haz_diferido_pasa_el_renglon_anterior():
    class Haz:
        lmn = object()
        vistos = []

        def decodificar(self, v, ii, chars, r, ant=""):
            self.vistos.append((r[0], ant))
            return r

    class Falso(completo.LectorCTC):
        def __init__(self):
            self.chars = list("ab")
            self.haz = Haz()

    l = Falso()
    v = np.array([[-0.01, -9.0]], np.float32)
    ii = np.array([[1, 0]], np.int64)
    out = [l._decodificar(v, ii, 0.0, 0), l._decodificar(v, ii, 0.0, 1)]
    assert Haz.vistos == []  # diferido
    l._cerrar_haz(out)
    assert Haz.vistos == [("a", ""), ("a", "a")]


def test_sha256_del_catalogo_rechaza_un_fichero_distinto(tmp_path):
    import pytest

    from amanuensis import modelos

    origen = tmp_path / "origen"
    origen.mkdir()
    (origen / "a.bin").write_bytes(b"hola")
    m = {"carpeta": "x", "ficheros": {"a.bin": 4}, "sha256": {"a.bin": "0" * 64}}
    with pytest.raises(OSError):
        modelos._bajar_ficheros(m, tmp_path / "destino", origen.parent.as_uri() + "/origen/..")  # base cualquiera: la ruta no existe
    import hashlib

    bueno = hashlib.sha256(b"hola").hexdigest()
    (origen / "x").mkdir()
    (origen / "x" / "a.bin").write_bytes(b"hola")
    m["sha256"]["a.bin"] = bueno
    modelos._bajar_ficheros(m, tmp_path / "d2", origen.as_uri())
    assert (tmp_path / "d2" / "a.bin").read_bytes() == b"hola"
    m["sha256"]["a.bin"] = "1" * 64
    with pytest.raises(OSError):
        modelos._bajar_ficheros(m, tmp_path / "d3", origen.as_uri())
