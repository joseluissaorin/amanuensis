"""amanuensis: OCR, HTR y voz sobre EmbeddingGemma 2.

    import amanuensis
    doc = amanuensis.leer("documento.pdf")          # Documento
    print(doc.exportar("md"))
    res = amanuensis.escuchar("audio.mp3")          # dict con texto y palabras con tiempos
"""
__version__ = "0.2.0"


def leer(ruta: str, **opciones):
    from .motor import Motor

    verbose = opciones.pop("verbose", False)
    paginas = opciones.pop("paginas", 0)
    return Motor(verbose=verbose, **opciones).leer(ruta, paginas)


def escuchar(ruta: str):
    import os

    from . import modelos
    from .modelos import entorno
    from .escucha import Escucha

    return Escucha(entorno("VOZ") or modelos.ruta("voz")).transcribir(ruta)
