# Amanuensis

**Lee cualquier página de la historia y escribe lo que le dictan. En tu ordenador, y sin inventar una sola letra.**

*Reads any page in history and writes down what it is told. On your own machine, without making up a single letter.* ([English below](#english))

Amanuensis es un lector especializado en **documentos históricos y en español**: manuscritos e impresos de los
siglos XII al XX, con sus abreviaturas y su ortografía, en una máquina corriente y sin mandar nada fuera. Lee también
impreso moderno y otras escrituras, pero no compite con los grandes modelos generales en lo que ellos hacen mejor (letra
a mano actual, documentos modernos complejos); las cifras de dónde gana y dónde pierde están abajo y en el informe técnico.

Un amanuense escribía lo que leía y lo que le dictaban. Amanuensis hace las dos cosas: OCR de impreso moderno
y antiguo, HTR de manuscritos de los siglos XII al XX y voz a texto con el instante de cada palabra, sobre el
codificador de EmbeddingGemma 2 de Google y unos lectores diminutos propios. No genera texto: reconoce. Cada
carácter sale con su posición en la imagen y su confianza.

```bash
uv tool install git+https://github.com/joseluissaorin/amanuensis
amanuensis documento.pdf                    # Markdown por la salida estándar
```

## La familia

| | qué es | dónde corre | para qué |
|---|---|---|---|
| **Amanuensis** | el modelo grande: la página entera (renglones, maquetación con tablas y notas al margen, orden de lectura, lector grande con selector por escritura) | GPU del Mac (Core ML) o CPU; ~7 s por página en un Mac | manuscritos, impresos antiguos, prensa, tablas, otras escrituras |
| **Amanuensis Ligero** | lector diminuto de 3,5 M parámetros; `--rapido` usa por dentro el Mínimo (1,07 M) | CPU, décimas de segundo por página | impreso latino de una columna, lotes grandes |
| **Amanuensis Voz** | voz a texto, módulo opcional | CPU, 37-45 veces el tiempo real | entrevistas, clases, subtítulos |

Por defecto (`-m auto`) cada página va a Amanuensis o a Ligero según lo que sea.

## Por qué

Examen sellado: 22 470 líneas de colecciones que no tocó ningún entrenamiento, en ocho grupos (impreso moderno y
antiguo, manuscritos medievales y modernos, letra a mano de los ss. XX-XXI, prensa, formularios, escaneos degradados).
CER en %, caracteres mal leídos; los demás sistemas, con su mejor configuración pública en las mismas imágenes.

| | media de los 8 grupos | letra a mano moderna | escaneos degradados | manuscritos (CMMHWR) |
|---|---|---|---|---|
| **Amanuensis** | **14,8** | 14,9 | 11,5 | **9,94** |
| **Amanuensis Ligero** | 21,5 | 26,0 | **10,2** | 14,8 |
| Amanuensis Ligero, modo rápido | 24,2 | 30,3 | 12,2 | 17,6 |
| PP-OCRv6 medium (afinado por Kiessling) | 20,1 | 18,0 | 19,6 | 15,7 |
| Gemma 4 E2B | 29,0 | **9,60** | 23,4 | 45,2 |
| dots.ocr | 29,7 | 16,3 | 25,5 | 42,8 |
| Tesseract (modelo de cada lengua) | 43,4 | 53,0 | 30,7 | 64,8 |
| Indra OCR | 55,7 | 42,2 | 59,9 | 73,9 |

Donde perdemos: la letra a mano moderna la leen mejor los modelos de visión y lenguaje grandes (Gemma 4: 9,6), y en
páginas enteras de prensa a muchas columnas el orden de lectura todavía intercala columnas (Tesseract nos gana ahí).
Detalle en el informe técnico.

## Qué hace

- **Documentos**: PDF, imágenes (también TIFF de varias páginas) o carpetas enteras a **Markdown, HTML, texto, JSON,
  TEI, PAGE XML, ALTO o hOCR**.
- **PDF digitales**: si el PDF trae una capa de texto fiable, la usa tal cual (instantáneo y exacto); solo hace OCR
  de las páginas sin capa o con la capa rota.
- **Manuscritos**: transcripción grafemática con los signos MUFI (ſ, ꝑ, ꝯ, ⁊…), como CATMuS.
- **Todas las escrituras** en impreso moderno con el lector multilingüe (`-m multilingue`).
- **Imágenes** (con la maquetación completa, `--imagenes`): figuras, grabados, capitulares, sellos y fórmulas
  recortados a la resolución original (`img/<documento>/p001_fig01.webp`), sin duplicar ornamentos repetidos,
  enlazados con su pie, con las llamadas «fig. N» del texto y la capitular con su palabra, y colocados en todas
  las salidas: Markdown `![pie](…)`, HTML `<figure>`, TEI `<figure><graphic/><head/><figDesc/>`, PAGE
  `ImageRegion`/`GraphicRegion`, ALTO `<Illustration>` y JSON. El texto alternativo es el pie o un descriptor
  Iconclass recuperado por similitud (no generado).
- **Página entera** con Amanuensis (`-m amanuensis`, o solo cuando hace falta con `auto`): renglones de alta
  resolución, maquetación con zonas, notas al margen, cabeceras, notas al pie, **tablas por celdas** (Markdown,
  HTML, TEI `<table>`, PAGE `TableRegion`, ALTO), orden de lectura aprendido y selector por escritura (alfabeto
  latino al lector histórico; árabe, cirílico, devanagari, CJK… al multilingüe).
- **Letras dudosas** marcadas con su probabilidad en todas las salidas: TEI `<unclear cert>`, hOCR `x_confs`, ALTO
  `WC`/`CC`, PAGE `conf`, HTML resaltado y JSON `dudas` (`--dudas P` fija el umbral).
- **Arrojo** (`--arrojo`, de 0 a 1; 0,5 = neutro): cuánto se arriesga el lector. Súbelo en escaneos muy malos o de
  baja resolución (calla menos letras); bájalo si prefieres huecos a inventos.
- **Estructura rica** (`--estructura diccionario|comic`): entradas de diccionario en TEI `<entry>`/`<form>`/
  `<gramGrp>`/`<sense>` y viñetas de cómic en `<div type="panel">`, también en Markdown y JSON.
- **Voz**: audio o vídeo a **texto, SRT, VTT o JSON** con el inicio y el fin de cada palabra, en 40 lenguas, con
  mayúsculas y puntuación y, si quieres, un modelo de lengua por idioma (`pip install "amanuensis[voz-lm]"`).
- **Orden de lectura aprendido** (un perceptrón diminuto por pares de líneas, 27 KB) y **páginas derechas**:
  corrige la inclinación y los giros de 90, 180 y 270 grados de escaneos y fotos.
- **Local**: nada sale de tu máquina. Los pesos (de 4 MB a 350 MB) se descargan solos la primera vez.

## Después de leer: el texto de lectura

Amanuensis no se queda en lo que el lector ve letra a letra. Después de leer pule la página y corrige lo que la
página misma permite corregir, sin tocar el modelo y sin inventar: cada cambio tiene que salir de la geometría, de
las series del propio documento o de lo que el lector vio con su probabilidad.

- **Texto de lectura por defecto.** Markdown, TXT y HTML salen sin folios, cabeceras corrientes, signaturas, marcas
  de página ni restos del escaneo, con los párrafos y las palabras partidas unidos también de una página a otra (y
  entre las dos hojas de una doble página) y con el verso en sus renglones. Las notas van al final. `--fiel` da la
  página tal cual, con el mobiliario en comentarios; TEI, PAGE, ALTO, hOCR y JSON lo conservan siempre todo.
- **Dobles páginas.** Un libro abierto escaneado en una sola imagen se parte por el canal y se lee hoja a hoja; las
  coordenadas vuelven a la página entera.
- **Renglones curvos.** En una fotocopia de libro abierto los renglones suben a lo ancho de la página: se siguen tira a
  tira, se enderezan y se releen, y gana la lectura más segura.
- **Prensa y revistas a varias columnas.** Las columnas se reconocen por los comienzos de renglón alineados; los
  trozos de un renglón que el detector partió (un subrayado a lápiz) se unen si entre ellos hay tinta, las cajas que
  cruzan el canal se parten por su franja blanca y la página se ordena por columnas. Si un bloque acaba en palabra
  partida («para el comba-»), lo sigue el bloque que la completa («te). En la posición»).
- **Pulido de la página** (sin maquetación): cabeceras repetidas y folios fuera del texto, epígrafes numerados y en
  versalitas con su nivel, títulos de capítulo, figuras («ESQUEMA 5.1.» y sus rótulos), filas partidas unidas («(1)»
  con su ejemplo), basura del escaneo.
- **Series del documento** (en todas las páginas): romanos y letras de una enumeración por su puesto («m)» entre II y
  IV es III), ejemplos «(n)» correlativos, llamadas «()» y notas pegadas a la palabra («bebedeira2»), números de verso,
  numeración jerárquica de los epígrafes («1.» entre 1.4.2 y 1.6 es 1.5), nombres de personaje al margen.
- **Signos.** Las comillas que el lector ve pero calla (el blanco les gana por poco) se escriben si pasan del 8 % de
  probabilidad en su hueco, y la que le falta a una pareja se pone donde el lector le dé más probabilidad. La raya
  media y la larga se deciden midiendo el trazo en la imagen, con los anchos del propio documento; ⊢ se reconoce por
  su forma (Ligero no lo tiene en su juego de caracteres). La notación propia del documento («+>», «>>») devuelve a
  su forma las lecturas rotas.
- **Diacríticos imposibles.** Un diacrítico que no existe en ninguna lengua del documento es una lectura falsa de otro
  (en portugués y en tupí, «Tupā» es «Tupã»; en español, «despuès» es «después»); en letón o en alemán, donde sí
  existe, no se toca. Los textos antiguos (ſ, ꝑ…) no pasan por aquí.
- **Corrección léxica guiada por el lector, en 31 lenguas.** Una palabra dudosa solo se cambia por otra que exista y
  que el lector vea casi igual de bien en sus propias probabilidades (la CTC) en ese tramo del renglón; además aprende
  las confusiones de cada libro («ti» leída «u», «ri» leída «n») y usa como diccionario lo que el propio documento lee
  con seguridad. La lengua se detecta en cada página. Las lenguas sin léxico (el tupí) y los textos antiguos solo se
  corrigen con ese vocabulario propio.

Opciones: `--pulido no`, `--lexico auto|local|no` (`auto`, por defecto, construye la primera vez el léxico de una
lengua que haga falta; `local`, solo los que ya haya), `--rectificar no`, `--fiel`. Desde Python, las mismas:
`amanuensis.leer(ruta, pulido=True, lexico="auto", rectificar=True, fiel=False)`.

## Léxicos

Uno por lengua para las 31 de Yappy (ar, bg, cs, da, de, el, en, es, et, fi, fr, hi, hr, hu, id, it, ja, ko, lt, lv, nl,
pl, pt, ro, ru, sk, sl, sv, tr, uk, vi), en `~/.cache/amanuensis/lexicos/` (unos 10 MB cada uno). Cada uno suma las
frecuencias de los subtítulos de OpenSubtitles (FrequencyWords 2018, contenido CC BY-SA 4.0) y, en 14 lenguas, las de
la Wikipedia (wikipedia-word-frequency, texto CC BY-SA), las dos fuentes llevadas a la misma escala. La detección de
lengua viaja con el paquete (`perfiles.json`) y no necesita los léxicos. El japonés no se corrige palabra a palabra
(no separa palabras con espacios).

```bash
amanu lexicos                      # cuáles hay y cuáles faltan
amanu lexicos construir es pt      # construirlos antes (por ejemplo, al instalar una app)
amanu lexicos construir --todas    # los 31 (unos 330 MB)
```

## Instalar

```bash
uv tool install git+https://github.com/joseluissaorin/amanuensis    # recomendado
pipx install git+https://github.com/joseluissaorin/amanuensis       # o con pipx
pip install git+https://github.com/joseluissaorin/amanuensis        # o en tu entorno
```

Python 3.10 o más. Sin PyTorch: solo ONNX Runtime. Con GPU de NVIDIA (CUDA 12): `pip install "amanuensis[gpu]"`
(onnxruntime-gpu anterior a la 1.23 y las bibliotecas de CUDA 12 de pip; las versiones nuevas de onnxruntime-gpu piden
CUDA 13 y, si no lo encuentran, se quedan en la CPU sin avisar). Para audio que no sea WAV hace falta `ffmpeg`.
Órdenes equivalentes: `amanuensis`, `amanu` y `eg2ocr` (el nombre antiguo).

## Usar

```bash
amanu documento.pdf                           # Markdown
amanu pagina.jpg -f tei -o salida/            # salida/pagina.tei.xml
amanu carpeta/ -f md,page,alto,json -o x/     # lote: todos los PDF e imágenes, varios formatos
amanu escaneos/ -m ligero -o x/               # Amanuensis Ligero: CPU, décimas por página
amanu escaneos/ -r -o x/                      # Ligero en modo rápido (lector Mínimo)
amanu folio.png -m amanuensis -f tei,page -o x/  # siempre el modelo grande: maquetación, tablas, notas al margen
amanu fax.png --arrojo 0.8                    # escaneo muy malo: el lector se arriesga más
amanu tomo.pdf --estructura diccionario -f tei -o x/   # entradas de diccionario en TEI
amanu libro.pdf --modo ocr --paginas 10       # ignorar la capa de texto; solo 10 páginas
amanu escucha entrevista.mp3 -f srt,txt -o subtitulos/
amanu libro.pdf --fiel                        # página a página, con folios y cabeceras en comentarios
amanu libro.pdf --lexico no --pulido no       # la lectura tal cual sale del lector
amanu modelos                                 # catálogo y qué hay descargado
amanu lexicos                                 # léxicos de las 31 lenguas
amanu visor                                   # visor local en http://127.0.0.1:8766/
```

Modelos (`-m`): `auto` (por defecto: una sonda de 0,5 s deja en Ligero el impreso moderno latino de una columna y
manda lo demás a Amanuensis), `amanuensis`, `ligero` (con `-r`, modo rápido). Siguen valiendo los nombres antiguos:
`completo` = `amanuensis`, `minimo` = `ligero -r`. Para expertos: `historico`, `multilingue`, `impreso`. En el Mac,
Amanuensis usa Core ML (GPU) si está `coremltools` (`pip install "amanuensis[coreml]"`).

Desde Python:

```python
import amanuensis
doc = amanuensis.leer("documento.pdf", arrojo=0.5, umbral_duda=0.5)
print(doc.exportar("md"))            # texto de lectura; "txt", "html", "json", "tei", "page", "alto", "hocr"
doc.fiel = True                      # md, txt y html página a página, con el mobiliario en comentarios
voz = amanuensis.escuchar("audio.mp3")
print(voz["texto"], voz["palabras"][0])  # {"texto", "inicio", "fin", "conf"}
```

En una app, crea el motor una vez y reutilízalo (los modelos y los léxicos se cargan una sola vez por proceso), y
prepara los léxicos de las lenguas que esperes para no esperar en la primera lectura:

```python
from amanuensis.motor import Motor
from amanuensis import lexico

lexico.preparar(["es", "pt"])                  # los construye si faltan; devuelve los que hay
motor = Motor("auto", verbose=False)            # pulido=True, lexico="auto", rectificar=True
doc = motor.leer("libro.pdf")
md = doc.exportar("md")
```

## Modelos

| nombre | qué es | tamaño | licencia de uso |
|---|---|---|---|
| `amanuensis` | Amanuensis: página entera (ONNX de 8 bits; Core ML opcional en el Mac) | 500 MB (+ 670 MB Core ML) | no comercial* |
| `ligero` | Amanuensis Ligero (3,5 M; con `-r`, el Mínimo de 1,07 M) | 14 MB + 4 MB | no comercial* |
| `voz` | Amanuensis Voz: 40 lenguas, mayúsculas y puntuación, tiempos por palabra | 460 MB | Apache 2.0 |
| (`voz-lm`) | modelos de lengua por idioma para la voz (37), bajo demanda | 12-460 MB cada uno | CC BY-SA 4.0 |

`amanu modelos --todos` lista también las piezas internas (detector Mínimo, lectores histórico y multilingüe, que
Amanuensis baja solo cuando le hacen falta) y las versiones anteriores.

\* Parte de los datos de entrenamiento (HSMS, OHG) es CC BY-NC 4.0. Detalles, datos y licencias de origen en
[MODEL_CARD.md](MODEL_CARD.md).

Dentro del paquete viajan además el modelo de continuidad del texto (`continuidad.npz`, uso no comercial, con la
atribución CC BY-SA de la Wikipedia) y los perfiles de lengua (`perfiles.json`, CC BY-SA 4.0); ver [NOTICE](NOTICE).

## Límites conocidos

- Prensa a muchas columnas: el orden de lectura aprendido todavía intercala columnas en páginas de cientos de
  líneas. La letra manuscrita moderna (s. XX-XXI) se lee mal.
- Ligero y su modo rápido segmentan mal los manuscritos: para ellos, Amanuensis.
- La transcripción de manuscritos es grafemática: no desarrolla abreviaturas.
- Velocidad de Amanuensis: unos 7 s por página en un Mac con GPU (Core ML), 11 s solo con la CPU del Mac y 35-40 s
  en una CPU x86 antigua; Ligero, décimas (con el rectificado y el léxico, unas décimas más por página; la carga de un
  léxico, unos 3 s la primera vez en cada proceso).
- La corrección léxica actúa sobre los renglones de Ligero (los del modelo grande no guardan sus probabilidades) y el
  pulido de estructura, en las páginas sin maquetación; las series, en todas.
- Ligero no tiene en su juego de caracteres algunos signos: ŷ, Î, Ŷ, Ỹ (tupí), ⊨, ⇒ (lógica) y los superíndices de ⁴
  en adelante; ⊢ se recupera por su forma. La cursiva no se reconoce.

## Variables de entorno

`AMANUENSIS_CACHE` (pesos; por defecto `~/.cache/amanuensis`), `AMANUENSIS_ORIGEN` (otro servidor de pesos),
`AMANUENSIS_MODELOS` (carpeta local con los modelos), `AMANUENSIS_VOZ` (carpeta de un modelo de voz),
`AMANUENSIS_VOZ_LM` (`auto`, `si`, `no`), `AMANUENSIS_VOZ_LENGUA`, `AMANUENSIS_COREML=0` (sin Core ML),
`AMANUENSIS_PULIDO=0`, `AMANUENSIS_LEXICOS` (`auto`, `local`, `no`), `AMANUENSIS_RECTIFICAR=0` (lo mismo que las
opciones), `AMANUENSIS_SIGNOS` (umbral de las comillas tímidas, 0,08; 0 las apaga), `AMANUENSIS_COLUMNAS=0` (sin la
maquetación de columnas), `AMANUENSIS_ORDEN` (orden de lectura; por defecto `prensa`: el orden por grafo disperso, `orden_gcn.npz`, en
las páginas a dos columnas o más y el perceptrón por pares en lo demás; `gcn` = el grafo en todas las páginas y
tamaños, también la maqueta mínima y el modelo grande; `ligero` = solo el perceptrón; `columnas` = corte XY con canales;
sin pesos del grafo, el orden de siempre),
`AMANUENSIS_CONTINUIDAD_BLOQUES=0` (sin reordenar bloques de columna por la palabra partida) y `AMANUENSIS_DEPURAR_LEXICO=1` (escribe por la salida de errores cada corrección léxica con su puntuación).

---

## English

Amanuensis does what a scribe did: it writes down what it reads and what it is told. OCR for modern and early
print, handwritten text recognition for manuscripts from the 12th to the 20th century, and speech to text with the
start and end time of every word, on top of Google's EmbeddingGemma 2 encoder plus our own tiny readers. It does
not generate text; it recognizes it. Every character comes with its position in the image and its confidence.

```bash
uv tool install git+https://github.com/joseluissaorin/amanuensis
amanuensis document.pdf                      # Markdown to stdout
amanu folio.jpg -f tei,page,alto -o out/     # TEI, PAGE XML and ALTO
amanu scans/ -m ligero -o out/               # Amanuensis Ligero: tiny CPU reader (-r: even faster)
amanu escucha interview.mp3 -f srt -o subs/  # speech to subtitles with word timings
```

- **Sealed exam** (22,470 lines never seen in training, 8 groups): Amanuensis 14.8 % mean CER, Ligero 21.5 %,
  PP-OCRv6 medium 20.1 %, Gemma 4 E2B 29.0 %, dots.ocr 29.7 %, Tesseract 43.4 %, Indra OCR 55.7 %. We lose on
  modern handwriting (Gemma 4: 9.6 vs 14.9) and on reading order in many-column newspapers.
- **The family**: Amanuensis (the large whole-page model), Amanuensis Ligero (3.5 M parameters, CPU; `-r` uses the
  1.07 M Mínimo reader) and Amanuensis Voz (optional speech module). `-m auto` picks Amanuensis or Ligero per page.
- **Whole pages** with Amanuensis (`-m amanuensis`): high-resolution baselines, layout zones, marginal notes,
  tables cell by cell, learned reading order and a script selector (Latin to the historical reader, other scripts
  to the multilingual one); ONNX Runtime with 8-bit weights, Core ML on Macs.
- **Doubtful letters** carry their probability in every output (TEI `<unclear cert>`, hOCR, ALTO, PAGE, HTML, JSON);
  `--arrojo` (0-1, 0.5 neutral) makes the reader bolder on very poor scans or more cautious on clean ones.
- **Reading text by default**: Markdown, plain text and HTML come without page numbers, running heads or page breaks,
  with paragraphs and hyphenated words joined across pages and verse kept line by line (`--fiel` for the page as it
  is). Double-page scans are split at the gutter; curved lines are straightened and re-read.
- **After reading**: the page is tidied (running heads, numbered headings, figures), numbered series are inferred
  from the document itself (list numerals, examples, footnote calls, verse numbers, heading numbering) and doubtful
  words are corrected with a CTC-guided lexical pass in 31 languages, detected page by page; languages without a
  lexicon and early texts use only the document's own confident vocabulary. `amanu lexicos` lists and builds the
  lexicons (OpenSubtitles and Wikipedia frequency lists, CC BY-SA).
- **Reading order**: pages with two or more columns are ordered by a sparse-graph network over the line boxes
  (a three-column theatre magazine: 12.6 → 6.2 % WER against Gemini; single-column books unchanged).
  `AMANUENSIS_ORDEN=gcn|prensa|ligero|columnas` picks the model for every size (line detector, newspaper columns,
  minimal layout and the large model): `prensa` (default) uses the graph only on multi-column pages, `gcn` everywhere,
  `ligero` the pairwise perceptron, `columnas` an XY cut with column gutters.
- **Speech**: 40 languages with casing and punctuation, optional per-language KenLM.
- **Digital PDFs** use their own text layer when it passes sanity checks; only the missing pages are OCRed.
- **Outputs**: Markdown, plain text, JSON, TEI, PAGE XML, ALTO, hOCR; SRT, VTT and JSON for speech.
- **Local and private**: no cloud, no GPU needed. Weights download on first use.
- **Licences**: code Apache 2.0. Some weights are non-commercial because part of their training data is
  CC BY-NC 4.0; see [MODEL_CARD.md](MODEL_CARD.md). The bundled `continuidad.npz` is non-commercial and
  `perfiles.json` is CC BY-SA 4.0; see [NOTICE](NOTICE).

Author: José Luis Saorín Ferrer. Demo in the browser: https://lector-eg2.pages.dev
