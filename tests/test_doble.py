"""Las mismas pruebas que crates/yappy-core/src/ocr/doble.rs (Yappy), más el mobiliario de la salida Markdown."""

import numpy as np
from PIL import Image

from amanuensis import doble
from amanuensis.salidas import Documento, Linea, Pagina, es_letras_esparcidas


def renglones(m, x0, x1):
    """Una «hoja» de renglones: franjas negras de 3 px cada 8 px, con huecos entre palabras."""
    alto = m.shape[0]
    for y in range(40, alto - 40, 8):
        for x in range(x0, x1):
            if (x // 23) % 5 != 4:
                m[y:y + 3, x] = 0


def pagina(w, h, *hojas):
    m = np.full((h, w), 255, np.uint8)
    for a, b in hojas:
        renglones(m, a, b)
    return m


def test_la_doble_pagina_se_parte_por_el_canal():
    img = Image.fromarray(pagina(1600, 1000, (80, 740), (860, 1520)))
    x = doble.canal(img)
    assert x is not None and 740 <= x <= 860
    _, i, d = doble.partir(img)
    assert i.size[0] + d.size[0] == 1600


def test_la_sombra_del_lomo_tambien_es_canal():
    m = pagina(1600, 1000, (80, 740), (860, 1520))
    m[:, 770:830] = 30
    assert doble.canal(Image.fromarray(m)) is not None


def test_el_texto_que_cruza_el_centro_no_se_parte():
    assert doble.canal(Image.fromarray(pagina(1600, 1000, (80, 1520)))) is None


def test_una_pagina_vertical_no_se_parte():
    assert doble.canal(Image.fromarray(pagina(1000, 1400, (60, 460), (540, 940)))) is None


def test_con_una_hoja_en_blanco_no_hay_nada_que_separar():
    assert doble.canal(Image.fromarray(pagina(1600, 1000, (860, 1520)))) is None


def test_una_pagina_girada_no_se_parte():
    # una página de texto corrido escaneada de lado: apaisada, con los renglones en vertical
    m = pagina(1000, 1600, (80, 920)).T.copy()
    assert doble.canal(Image.fromarray(m)) is None


def test_una_pagina_girada_con_un_blanco_en_el_centro_no_se_parte():
    # la misma, con un blanco entre dos párrafos a media página: girada, el blanco cae donde iría el canal
    m = pagina(1000, 1600, (80, 920))
    m[730:870] = 255
    assert doble.canal(Image.fromarray(m.T.copy())) is None
    assert doble.parece_vertical(Image.fromarray(m.T.copy()))
    assert not doble.parece_vertical(Image.fromarray(pagina(1600, 1000, (80, 740), (860, 1520))))


def test_letras_esparcidas():
    assert es_letras_esparcidas("L A C O M U N I C A C I Ó N")
    assert es_letras_esparcidas("LA C O M U N I CA C I Ó &")
    assert not es_letras_esparcidas("1.1. EN BUSCA DE REGULARIDADES")
    assert not es_letras_esparcidas("CAPÍTULO IV")
    assert not es_letras_esparcidas("La comunicación")


def test_el_folio_sale_del_texto_y_el_parrafo_se_une():
    ls = [
        Linea([100, 40, 600, 70], "L A C O M U N I C A C I Ó N"),
        Linea([100, 200, 900, 230], "Los ejemplos anteriores han puesto de relieve que el conocimiento"),
        Linea([880, 60, 920, 90], "16"),
        Linea([100, 240, 900, 270], "lingüístico asociado al código no es suficiente."),
    ]
    fiel = Documento("x", [Pagina(1000, 1400, ls)], fiel=True).exportar("md")
    assert "<!-- cabecera: L A C O M U N I C A C I Ó N -->" in fiel
    assert "<!-- numero: 16 -->" in fiel
    assert "conocimiento lingüístico asociado" in fiel
    # por defecto, texto de lectura: sin mobiliario
    md = Documento("x", [Pagina(1000, 1400, ls)]).exportar("md")
    assert "<!--" not in md and "16" not in md and "conocimiento lingüístico asociado" in md
