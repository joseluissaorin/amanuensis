"""La regla por página de la maquetación (AMANUENSIS_MAQUETA_MINIMA=pagina): la clásica en el impreso moderno limpio y la
mínima en lo demás; «0» y «auto» siguen como estaban."""
import pytest

from amanuensis import motor as M
from amanuensis.salidas import Linea, Pagina


def _lineas(confs):
    return [Linea([0, 20 * i, 400, 20 * i + 14], "texto de prueba", c, []) for i, c in enumerate(confs)]


def _motor(monkeypatch, confs, modo):
    monkeypatch.setenv("AMANUENSIS_MAQUETA_MINIMA", modo)
    monkeypatch.setattr(M.modelos, "disponible", lambda n: True)
    m = M.Motor.__new__(M.Motor)
    m.con_completo, m.repo, m.modelo, m.verbose = False, None, "ligero", False
    llamadas = []

    def clasica(img, t0):
        llamadas.append("clasica")
        return Pagina(400, 600, _lineas(confs), "ocr (detector)", "ligero")

    def minima(img, t0):
        llamadas.append("minima")
        return Pagina(400, 600, _lineas(confs), "ocr (maqueta mínima)", "ligero")

    m._pagina_clasica, m._pagina_minima = clasica, minima
    m.maqueta_minima = lambda para="ligero": object() if modo != "0" else None
    return m, llamadas


def test_es_moderno():
    assert M.Motor.es_moderno(_lineas([0.99] * 20))
    assert M.Motor.es_moderno(_lineas([0.99] * 18 + [0.5] * 2))  # un 10 % de renglones flojos: aún la clásica
    assert not M.Motor.es_moderno(_lineas([0.99] * 15 + [0.5] * 5))  # renglones flojos: manuscrito, gótica, foto
    assert not M.Motor.es_moderno(_lineas([0.99] * 2))  # muy pocos renglones


@pytest.mark.parametrize("confs,esperado", [([0.99] * 20, ["clasica"]), ([0.7] * 20, ["clasica", "minima"])])
def test_regla_por_pagina(monkeypatch, confs, esperado):
    m, llamadas = _motor(monkeypatch, confs, "pagina")
    p = m.ocr_paginas([None], partir=False)[0]
    assert llamadas == esperado
    assert ("por regla de página" in p.fuente) == (esperado[-1] == "minima")


@pytest.mark.parametrize("modo,esperado", [("auto", ["minima"]), ("0", ["clasica"])])
def test_modos_de_siempre(monkeypatch, modo, esperado):
    m, llamadas = _motor(monkeypatch, [0.7] * 20, modo)
    m.ocr_paginas([None], partir=False)
    assert llamadas == esperado


def test_por_defecto_es_la_regla(monkeypatch):
    monkeypatch.delenv("AMANUENSIS_MAQUETA_MINIMA", raising=False)
    monkeypatch.setattr(M.modelos, "disponible", lambda n: True)
    m = M.Motor.__new__(M.Motor)
    assert m.maqueta_por_pagina()
