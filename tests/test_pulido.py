"""El pulido de la página, la corrección léxica guiada por la CTC y la rectificación por tiras."""

import math

import numpy as np
from PIL import Image

from amanuensis import lexico, pulido, rectificado
from amanuensis.salidas import Documento, Linea, Pagina


def L(x0, y0, x1, y1, t, conf=0.99):
    return Linea([x0, y0, x1, y1], t, conf, confs=[conf] * len(t))


def test_la_fila_partida_se_une():
    ls = pulido.fusionar_filas([L(320, 100, 395, 140, "(1)"), L(417, 100, 920, 140, "Era pobre, pero honrado")])
    assert [l.texto for l in ls] == ["(1) Era pobre, pero honrado"]


def test_los_romanos_de_una_serie_van_en_su_puesto():
    ls = [L(225, 100, 900, 130, "I) la información contenida en el enunciado;"),
          L(225, 132, 900, 162, "I) los factores que configuran"),
          L(166, 164, 900, 194, "y la situación"),
          L(218, 196, 900, 226, "III) los principios conversacionales.")]
    pulido.series_romanas(ls)
    assert [l.texto[:4] for l in ls] == ["I) l", "II) ", "y la", "III)"]


def test_la_serie_puede_empezar_a_medias_y_perder_el_parentesis():
    ls = [L(205, 100, 900, 130, ") Cantidad. Se relaciona"), L(205, 300, 900, 330, "II) Cualidad. Esta categoría"),
          L(205, 500, 900, 530, "m) Relación. Contiene una única máxima"), L(205, 700, 900, 730, "IV) Modalidad. Se relaciona")]
    pulido.series_romanas(ls)
    assert [l.texto.split()[0] for l in ls] == ["I)", "II)", "III)", "IV)"]


def test_letras_de_una_lista():
    ls = [L(1384, 100, 1900, 130, "b. María tiene hijos"), L(1384, 132, 1900, 162, "C. #María tiene tres hijos")]
    pulido.series_letras(ls)
    assert ls[1].texto.startswith("c. ")


def test_los_ejemplos_rotos_se_numeran_entre_los_buenos():
    p = Pagina(1000, 1400, [L(200, 100, 900, 130, "(D) Era pobre, pero honrado"), L(200, 300, 900, 330, "( A: —¿A qué hora?"),
                            L(200, 500, 900, 530, "(3) A un profesor le piden")])
    pulido.ejemplos_numerados([p])
    assert [l.texto[:3] for l in p.lineas] == ["(1)", "(2)", "(3)"]


def test_cabecera_corriente_y_folio():
    p = Pagina(1600, 2400, [L(262, 188, 317, 226, "80"), L(660, 187, 1250, 222, "INTRODUCCIÓN A LA PRAGMÁTICA")]
               + [L(262, 300 + 50 * k, 1630, 340 + 50 * k, "texto corrido de la página que sigue y sigue") for k in range(10)])
    pulido.pulir(p, {pulido.clave("INTRODUCCIÓN A LA PRAGMÁTICA")})  # repetida en otras páginas del documento
    assert p.lineas[0].zona_clase == "NumberingZone" and p.lineas[1].zona_clase == "RunningTitleZone"
    assert p.lineas[2].zona_clase is None


def test_la_numeracion_de_los_epigrafes_se_rehace():
    hs = ["1.4.2. ¿Es necesario el código?", "1. ¿Con quién nos comunicamos?", "1.6. ¿Cuándo hay comunicación?",
          "1.7. ¿Qué comunicamos?", "18. ¿Para qué nos comunicamos?", "1.9. ¿Por qué medio?"]
    p = Pagina(1000, 3000, [L(200, 300 * k, 900, 300 * k + 30, t) for k, t in enumerate(hs)])
    for l in p.lineas:
        l.zona_clase = "Section-header"
    pulido.numeracion_epigrafes([p])
    assert [l.texto.split()[0] for l in p.lineas] == ["1.4.2.", "1.5.", "1.6.", "1.7.", "1.8.", "1.9."]


def test_ctc_logp_de_una_secuencia_segura():
    chars = list("abc")
    lp = np.log(np.full((6, 4), 1e-6))
    for t, c in enumerate([1, 0, 2, 0, 3, 0]):
        lp[t, c] = 0.0
    assert lexico.ctc_logp(lp, [1, 2, 3]) > -1e-3
    assert lexico.ctc_logp(lp, [1, 3]) < -10


def test_confusiones_de_ocr():
    assert "prácticas" in lexico.confusiones("prácucas")
    assert "necesario" in lexico.confusiones("necesano")
    lex = {"prácticas": 900, "comunicativo": 900, "sentido": 900, "relativo": 900, "gramática": 900, "identidad": 900}
    sist = lexico.confusiones_del_documento(["prácucas", "comunicauvo", "senudo", "relauvo", "gramáuca", "idenudad"], lex)
    assert ("u", "ti") in sist
    # dos formas de la misma palabra no hacen una confusión del libro («proferencia», «proferencias»)
    lex = {"preferencia": 900, "preferencias": 900, "lastre": 900}
    assert ("o", "e") not in lexico.confusiones_del_documento(["proferencia", "proferencias", "lastro"], lex)


def test_el_espanol_antiguo_y_el_ingles_no_se_corrigen():
    lex = {w: 100 for w in "de la que el en y los del se las por un para con no una su al lo como mas pero texto".split()}
    def pag(texto):
        return Pagina(1000, 1000, [L(0, 20 * k, 900, 20 * k + 18, texto) for k in range(10)])
    assert lexico.es_espanol_moderno([pag("de la que el en y los del se las por un para con texto")], lex)
    assert not lexico.es_espanol_moderno([pag("the of and to in is that for it as with was on texto")], lex)


def test_la_rectificacion_endereza_un_renglon_inclinado():
    g = np.full((200, 800), 255, np.uint8)
    for x in range(40, 760):
        y = int(80 + 0.06 * x)  # sube 43 px a lo ancho: más de un cuerpo
        if (x // 9) % 4:
            g[y:y + 18, x] = 0
    img = Image.fromarray(g)
    r, desvio = rectificado.rectificar(img, [40, 85, 760, 115])
    assert r is not None and desvio > 1.0
    a = np.asarray(r) < 128
    filas = np.where(a.sum(1) > 0.3 * a.shape[1])[0]
    assert len(filas) and filas.max() - filas.min() < 26  # la tinta queda en una banda horizontal


def test_los_diacriticos_imposibles_en_portugues_y_tupi():
    from amanuensis.salidas import Pagina
    p = Pagina(1000, 1000, [L(0, 0, 900, 20, "Tupā o-mondyk Aimbirë gûasu ſeu"), L(0, 30, 900, 50, "lingüística")])
    pulido.diacriticos([p], ["pt", "es"])
    assert p.lineas[0].texto == "Tupã o-mondyk Aimbirê gûasu feu"
    assert p.lineas[1].texto == "lingüística"


def test_la_mayuscula_de_la_s_larga_no_es_una_f():
    from amanuensis.salidas import Pagina
    p = Pagina(1000, 1000, [L(0, 0, 900, 20, "Searle y Sobre la ſ")])
    pulido.diacriticos([p], ["es"])
    assert p.lineas[0].texto == "Searle y Sobre la f"


def test_en_leton_el_macron_es_una_letra():
    from amanuensis.salidas import Pagina
    p = Pagina(1000, 1000, [L(0, 0, 900, 20, "Rīga ir Latvijas galvaspilsēta")])
    pulido.diacriticos([p], ["lv"])
    assert p.lineas[0].texto == "Rīga ir Latvijas galvaspilsēta"


def test_las_llamadas_de_nota_pegadas():
    from amanuensis.salidas import Pagina
    p = Pagina(1000, 1000, [L(0, 20 * k, 900, 20 * k + 18, t) for k, t in
                            enumerate(["a bebedeira2, os", "das moçacaras3, e", "os feitiços4... e", "chorá-lo6, como"])])
    pulido.llamadas_de_nota([p])
    assert [l.texto for l in p.lineas] == ["a bebedeira^2, os", "das moçacaras^3, e", "os feitiços^4... e", "chorá-lo^6, como"]


def test_deteccion_de_lenguas():
    import re
    ws = lambda t: [w.lower() for w in re.findall(r"[^\W\d_]+", t)]
    if not lexico.ruta_lengua("pt"):
        return  # sin léxicos en la caché (python -m amanuensis.lexico construir)
    assert lexico.lenguas_de(ws("Fogo que nunca descansa, mas sempre causa dor, e com seu bravo furor tira toda a "
                                "esperança ao maldito pecador. Pecador, como te entregas aos vícios, tão sem freio")) == ["pt"]
    assert lexico.lenguas_de(ws("Karaiba na s-etá-î São Sebastião a e o-mondyk t-atá s-esé i mondýa N o-pytá-î amõ "
                                "abá maran-á-pe Îandé manhana ranhẽ t-o-só îandé r-enondé opá")) == []


def _prosa(y0, textos, x1=1500):
    return [L(100, y0 + 40 * k, x1, y0 + 40 * k + 30, t) for k, t in enumerate(textos)]


def test_la_exportacion_por_defecto_une_lo_que_corta_la_pagina():
    p1 = Pagina(1600, 2400, [L(700, 60, 900, 90, "80")] + _prosa(300, [
        "Primer renglón lleno de un párrafo que va a seguir en la otra página y no acaba aquí",
        "sino que continúa con la palabra partida al final de la comu-"]))
    p2 = Pagina(1600, 2400, [L(700, 60, 900, 90, "81")] + _prosa(300, [
        "nicación, que sigue en la página siguiente con su texto normal y corriente hasta el final.",
        "Y aquí termina el párrafo de verdad."], x1=1500))
    d = Documento("x", [p1, p2])
    for p in d.paginas:
        pulido.pulir(p, set())
    md = d.exportar("md")
    assert "comunicación, que sigue" in md and "80" not in md and "<!--" not in md and "página" not in md.split("\n")[0][:4]
    assert "\f" not in d.exportar("txt")
    d.fiel = True
    assert "<!-- página 2 -->" in d.exportar("md")


def test_el_verso_se_queda_en_sus_renglones():
    ls = [L(400, 300 + 40 * k, x1, 330 + 40 * k, t) for k, (t, x1) in enumerate(
        [("1169. Fogo que nunca descansa,", 1000), ("mas sempre causa dor,", 850), ("e com seu bravo furor", 900),
         ("tira toda a esperança", 880), ("ao maldito pecador.", 860)])]
    md = Documento("x", [Pagina(1748, 3000, ls)]).exportar("md")
    assert "descansa,  \nmas sempre" in md


def test_las_series_valen_tambien_en_paginas_con_maquetacion():
    ls = [L(225, 100, 900, 130, "I) la información contenida en el enunciado;"),
          L(225, 132, 900, 162, "I) los factores que configuran"),
          L(218, 196, 900, 226, "III) los principios conversacionales."),
          L(700, 10, 900, 40, "I) cabecera que no es texto")]
    for l in ls[:3]:
        l.zona_clase = "MainZone"
    ls[3].zona_clase = "RunningTitleZone"
    p = Pagina(1000, 1400, ls)
    pulido.pulir_texto(p)
    assert [l.texto[:4] for l in ls] == ["I) l", "II) ", "III)", "I) c"]


def test_en_un_texto_antiguo_graves_y_punto_medio_se_respetan():
    p = Pagina(1000, 1000, [L(0, 20 * k, 900, 20 * k + 18, "à la ſanta yglesia e·l rey dixo") for k in range(40)])
    assert pulido.lenguas_del_documento([p]) == []  # ſ por todas partes: antiguo
    pulido.diacriticos([p], pulido.lenguas_del_documento([p]))
    assert p.lineas[0].texto == "à la ſanta yglesia e·l rey dixo"


def test_en_espanol_moderno_el_grave_y_el_punto_medio_se_corrigen():
    p = Pagina(1000, 1000, [L(0, 0, 900, 20, "despuès de merecer·algún còdigo y la col·lecció")])
    pulido.diacriticos([p], ["es"])
    assert p.lineas[0].texto == "después de merecer algún código y la col·lecció"


def test_el_ejemplo_con_subletra_se_queda():
    p = Pagina(1000, 1000, [L(0, 0, 900, 20, "De (6a) debo inferir y de (9a, que no"), L(0, 30, 900, 50, "(6) a. Alicia")])
    pulido.referencias([p])
    assert p.lineas[0].texto == "De (6a) debo inferir y de (9)a, que no"


def test_la_deteccion_de_lenguas_no_necesita_los_lexicos(monkeypatch):
    import re
    monkeypatch.setattr(lexico, "ruta_lengua", lambda l: None)
    lexico.perfil.cache_clear()
    try:
        ws = [w.lower() for w in re.findall(r"[^\W\d_]+", "Die Würde des Menschen ist unantastbar. Sie zu achten und zu "
                                             "schützen ist Verpflichtung aller staatlichen Gewalt und das Volk")]
        assert lexico.lenguas_de(ws) == ["de"]
    finally:
        lexico.perfil.cache_clear()


def test_las_opciones_del_motor():
    from amanuensis.motor import Motor
    m = Motor("ligero", verbose=False, pulido=False, lexico="no", rectificar=False)
    assert (m.pulido, m.lexico, m.rectificar) == (False, "no", False)
    m = Motor("ligero", verbose=False)
    assert (m.pulido, m.lexico, m.rectificar) == (True, "auto", True)



def test_la_lista_arabiga_no_es_romana_y_recupera_su_signo():
    ls = [L(205, 100, 900, 130, "1) «Haga usted que su contribución»"), L(205, 300, 900, 330, "1 «No diga usted lo que crea falso»"),
          L(205, 500, 900, 530, "3) «Evite usted ser oscuro»")]
    pulido.pulir_texto(Pagina(1000, 1400, ls))
    assert [l.texto.split()[0] for l in ls] == ["1)", "1", "3)"]  # «1» en puesto de 2: no cuadra, no se toca
    ls = [L(205, 100, 900, 130, "1 «Haga usted que su contribución»"), L(205, 300, 900, 330, "2) «No diga usted»")]
    pulido.pulir_texto(Pagina(1000, 1400, ls))
    assert [l.texto.split()[0] for l in ls] == ["1)", "2)"]


def test_cabra_y_cabra_futuro_son_dos_palabras(tmp_path):
    r = tmp_path / "lex.txt"
    r.write_text("cabra 5000\ncabrá 150\nsiguiente 900000\nsiguíente 30\ncódigo 90000\ncodigo 800\n", encoding="utf-8")
    lx = lexico.cargar(str(r))
    assert "cabrá" in lx and "siguíente" not in lx and "codigo" not in lx
