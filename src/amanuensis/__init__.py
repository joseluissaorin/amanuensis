"""amanuensis: OCR, HTR y voz sobre EmbeddingGemma 2.

    import amanuensis
    doc = amanuensis.leer("documento.pdf")          # Documento
    print(doc.exportar("md"))
    res = amanuensis.escuchar("audio.mp3")          # dict con texto y palabras con tiempos

Familia: Amanuensis (modelo grande), Amanuensis Ligero (rapido=True: lector Mínimo) y Amanuensis Voz (escuchar).
Opciones de leer(): modelo («auto», «amanuensis», «ligero»; alias «completo» y «minimo»; expertos «historico»,
«multilingue», «impreso»), arrojo (0-1, 0,5 = neutro), umbral_duda (0,5), estructura («no», «auto», «diccionario»,
«comic»), rapido, imagenes, enderezar, modo («auto», «ocr», «texto»), dpi, paginas, y lo de después de leer:
pulido (True), lexico («auto», «local», «no»), rectificar (True) y fiel (False).

Por defecto, exportar("md" | "txt" | "html") da texto de lectura: sin folios, cabeceras ni marcas de página, con los
párrafos y las palabras partidas unidos también entre páginas y el verso en sus renglones. fiel=True (aquí o en
doc.fiel) da la página tal cual, con el mobiliario en comentarios. TEI, PAGE, ALTO, hOCR y JSON lo conservan todo.

Para una app: un Motor se crea una vez y se reutiliza (amanuensis.motor.Motor(...).leer(ruta)); los léxicos de una
lengua se construyen la primera vez que hacen falta (amanuensis.lexico.preparar(["es", "pt"]) los deja listos antes).
"""
__version__ = "1.1.0"


def leer(ruta: str, **opciones):
    from .motor import Motor

    verbose = opciones.pop("verbose", False)
    paginas = opciones.pop("paginas", 0)
    fiel = opciones.pop("fiel", False)
    d = Motor(verbose=verbose, **opciones).leer(ruta, paginas)
    d.fiel = fiel
    return d


def escuchar(ruta: str, lm: str | None = None, lengua: str | None = None, arrojo: float = 0.5, hilos: int = 0):
    from . import modelos
    from .modelos import entorno
    from .escucha import Escucha

    return Escucha(entorno("VOZ") or modelos.ruta("voz"), hilos, lm=lm, lengua=lengua, arrojo=arrojo).transcribir(ruta)
