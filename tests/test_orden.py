"""El enganche del orden por grafo (AMANUENSIS_ORDEN=gcn) con un modelo de mentira: se usa si hay pesos y, si no, cada
camino sigue con su orden de siempre."""
import sys
import types

import pytest

from amanuensis import detector

CAJAS = [[0, 0, 100, 10], [0, 20, 100, 30], [0, 40, 100, 50]]


def _stub(monkeypatch, modelo=True, orden=None):
    m = types.ModuleType("amanuensis.orden_gcn")
    m.modelo = lambda: ({"stub": 1} if modelo else None)
    m.ordenar = lambda m_, cajas, tam: list(orden if orden is not None else reversed(range(len(cajas))))
    monkeypatch.setitem(sys.modules, "amanuensis.orden_gcn", m)
    import amanuensis
    monkeypatch.setattr(amanuensis, "orden_gcn", m, raising=False)


def test_gcn_si_hay_pesos(monkeypatch):
    _stub(monkeypatch)
    monkeypatch.setenv("AMANUENSIS_ORDEN", "gcn")
    assert detector.orden_lectura(CAJAS, "aprendido", (100, 60)) == [2, 1, 0]
    assert detector.orden_lectura(CAJAS, "gcn", (100, 60)) == [2, 1, 0]


def test_sin_modo_no_se_usa(monkeypatch):
    _stub(monkeypatch)
    monkeypatch.delenv("AMANUENSIS_ORDEN", raising=False)
    assert detector.orden_lectura(CAJAS, "aprendido", (100, 60)) == [0, 1, 2]


@pytest.mark.parametrize("modelo,orden", [(False, None), (True, [0, 0, 1])])
def test_cae_limpio(monkeypatch, modelo, orden):
    # sin pesos, o con una salida que no es una permutación: el orden de siempre
    _stub(monkeypatch, modelo, orden)
    monkeypatch.setenv("AMANUENSIS_ORDEN", "gcn")
    assert detector.orden_gcn(CAJAS, (100, 60)) is None
    assert detector.orden_lectura(CAJAS, "aprendido", (100, 60)) == [0, 1, 2]
    assert detector.orden_lectura(CAJAS, "gcn", (100, 60)) == [0, 1, 2]


def test_modo_desconocido(monkeypatch):
    monkeypatch.setenv("AMANUENSIS_ORDEN", "otro")
    assert detector.modo_orden() == ""


def test_prensa_por_defecto(monkeypatch):
    # sin AMANUENSIS_ORDEN, el grafo solo en columnas.maquetar (páginas a varias columnas); el detector sigue con ligero
    from amanuensis import columnas
    _stub(monkeypatch)
    monkeypatch.delenv("AMANUENSIS_ORDEN", raising=False)
    llamadas = []
    orig = detector.orden_lectura
    monkeypatch.setattr(detector, "orden_lectura", lambda c, modo="aprendido", tam=None: llamadas.append(modo) or orig(c, modo, tam))
    monkeypatch.setattr(columnas, "comienzos", lambda cajas, h: [0.0, 500.0])
    monkeypatch.setattr(columnas, "unir_trozos", lambda cajas, cols, h, g=None: cajas)
    monkeypatch.setattr(columnas, "partir_por_canales", lambda img, cajas: (cajas, 0))
    from PIL import Image
    cajas = [[0, 20 * i, 100, 20 * i + 10] for i in range(10)]
    out, r = columnas.maquetar(Image.new("L", (1000, 300), 255), cajas)
    assert llamadas == ["gcn"] and out == cajas[::-1]
    monkeypatch.setenv("AMANUENSIS_ORDEN", "columnas")
    llamadas.clear()
    columnas.maquetar(Image.new("L", (1000, 300), 255), cajas)
    assert llamadas == ["columnas"]
