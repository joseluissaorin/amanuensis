import json

from amanuensis.salidas import Documento, Linea, Pagina


def _pagina(prensa=True):
    ls, zs, y = [], [], 0

    def bloque(textos, alto):
        nonlocal y
        z = len(zs)
        for t in textos:
            ls.append(Linea([100, y, 700, y + alto], t, zona=z, zona_clase="MainZone"))
            y += alto + 4
        zs.append({"clase": -1, "tipo": "MainZone", "caja": [100, y - 10, 700, y], "conf": 1.0, "prensa": True})
        y += 30

    bloque(["PRIMER TITULAR DE LA PÁGINA"], 60)
    bloque(["Texto de la primera noticia, con varias palabras en el renglón.", "Segundo renglón de la primera noticia aquí mismo."], 20)
    bloque(["OTRO TITULAR DISTINTO"], 60)
    bloque(["Texto de la segunda noticia, también con palabras suficientes.", "Y otro renglón más de la segunda noticia, claro."], 20)
    p = Pagina(800, 900, ls, "ocr (Amanuensis, página entera), prensa" if prensa else "ocr")
    p.zonas = zs
    return p


def test_articulos_en_json_y_separadores():
    d = Documento("x.jpg", [_pagina()])
    js = json.loads(d.exportar("json"))["paginas"][0]
    assert len(js["articulos"]) == 2 and js["articulos"][0]["titulo"].startswith("PRIMER")
    assert "---" in d.exportar("md")
    assert '<div type="article">' in d.exportar("tei")
    assert "---" not in d.exportar("txt")  # el txt de lectura no lleva separadores (no cuenta como texto al medir)


def test_sin_prensa_no_hay_articulos():
    d = Documento("x.jpg", [_pagina(prensa=False)])
    assert "articulos" not in json.loads(d.exportar("json"))["paginas"][0]


def test_articulos_de_la_red_sobreviven_a_los_bloques():
    d = Documento("x.jpg", [_pagina()])
    p = d.paginas[0]
    # la red agrupa por renglones: los dos primeros bloques (renglones 0-2) en un artículo y el resto en otro
    p._art_red = [{"lineas": p.lineas[:3], "titulo": "uno", "confianza": 0.9}, {"lineas": p.lineas[3:], "titulo": "dos", "confianza": 0.8}]
    js = json.loads(d.exportar("json"))["paginas"][0]
    assert [a["titulo"] for a in js["articulos"]] == ["uno", "dos"]
    assert sum(len(a["bloques"]) for a in js["articulos"]) == len(js["bloques"])
    assert "_art_red" not in js


def test_articulos_red_sin_tokenizers_avisa_y_usa_reglas(monkeypatch):
    from amanuensis.motor import Motor

    m = Motor.__new__(Motor)
    m._art_red, m.hilos, m.verbose = None, 1, False
    monkeypatch.setenv("AMANUENSIS_RUTA_ARTICULOS_RED", "/no/existe")
    p = _pagina()
    m._articulos_red(p, __import__("PIL.Image", fromlist=["Image"]).new("RGB", (100, 100)))
    assert m._art_red is False and not getattr(p, "_art_red", None)
