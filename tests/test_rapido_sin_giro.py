"""Modo rápido (sin enderezar): la maqueta mínima no gira la página con su clasificador de orientación."""
from amanuensis import motor as M



def test_rapido_sin_giro_de_la_minima(monkeypatch):
    # sin enderezar (modo rápido), la maqueta mínima no gira la página con su clasificador
    from PIL import Image
    pedido = {}

    class MM:
        def analizar(self, img, orientar=True, lado=None, puntuar=None):
            pedido["orientar"] = orientar
            return {"imagen": img, "lineas": [], "zonas": [], "giro": 0}

    m = M.Motor.__new__(M.Motor)
    m.enderezar, m.modelo, m.verbose = False, "minimo", False
    m.maqueta_minima = lambda para="ligero": MM()
    m.lector = lambda n: None
    m.leer_lineas = lambda *a, **k: []
    m.reordenar = lambda l, *a, **k: l
    m.lineas_giradas = lambda im, l, lector=None: l
    m.lineas_minima(Image.new("L", (40, 40), 255))
    assert pedido["orientar"] is False
