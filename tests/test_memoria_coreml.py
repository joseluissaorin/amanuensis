from amanuensis import completo_coreml as CM


def test_huella_positiva_en_mac():
    import platform

    if platform.system() == "Darwin":
        assert CM.huella_gb() > 0.01


def test_vaciar_suelta_funciones():
    f = CM._Funciones.__new__(CM._Funciones)
    f._f = {"c8": object(), "c16": object()}
    f.vaciar()
    assert not f._f
