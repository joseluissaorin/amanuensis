# Amanuensis

**Lee cualquier página de la historia y escribe lo que le dictan. En tu ordenador, y sin inventar una sola letra.**

*Reads any page in history and writes down what it is told. On your own machine, without making up a single letter.* ([English below](#english))

Un amanuense escribía lo que leía y lo que le dictaban. Amanuensis hace las dos cosas: OCR de impreso moderno
y antiguo, HTR de manuscritos de los siglos XII al XX y voz a texto con el instante de cada palabra, sobre el
codificador de EmbeddingGemma 2 de Google y unos lectores diminutos propios. No genera texto: reconoce. Cada
carácter sale con su posición en la imagen y su confianza.

```bash
uv tool install git+https://github.com/joseluissaorin/amanuensis
amanuensis documento.pdf                    # Markdown por la salida estándar
```

## Por qué

| | Indra OCR | Tesseract | **Amanuensis Mínimo** | **Amanuensis Ligero** |
|---|---|---|---|---|
| Parámetros del lector | 1,24 M | sin medir | **1,07 M** | 3,5 M |
| Impreso moderno, líneas (CER limpio / degradado) | 0,84 / 9,06 | 9,69 / 15,80 | **0,28 / 4,46** | **0,21 / 4,00** |
| Páginas con sus líneas (CER limpio / degradado) | 0,72 / 5,25 | sin medir | 0,43 / 3,22 | **0,38 / 2,69** |
| Manuscritos e impresos antiguos (CER de línea) | 76,1 | 74,7 | 28,1 | **24,9** |
| Páginas por segundo, CPU de un portátil (14 procesos) | 33,9 | sin medir | **34,0** | 19,1 |

CER en %: caracteres mal leídos. Bancos propios no vistos en el entrenamiento: 1000 líneas y 35 páginas de impreso
moderno, en limpio y degradadas, y 1214 líneas de manuscritos e impresos de los siglos XII-XVIII (latín y lenguas
romances). Todo medido en el mismo Mac (M4 Max), en las mismas condiciones. Con el lector grande (`-m historico`,
EmbeddingGemma 2 + adaptador LoRA) el error en manuscritos baja al 20 %.

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
- **Voz**: audio o vídeo a **texto, SRT, VTT o JSON** con el inicio y el fin de cada palabra (español, catalán,
  gallego, euskera y portugués; más lenguas en la próxima versión).
- **Local**: nada sale de tu máquina. Los pesos (de 4 MB a 350 MB) se descargan solos la primera vez.

## Instalar

```bash
uv tool install git+https://github.com/joseluissaorin/amanuensis    # recomendado
pipx install git+https://github.com/joseluissaorin/amanuensis       # o con pipx
pip install git+https://github.com/joseluissaorin/amanuensis        # o en tu entorno
```

Python 3.10 o más. Sin PyTorch: solo ONNX Runtime. Para audio que no sea WAV hace falta `ffmpeg`.
Órdenes equivalentes: `amanuensis`, `amanu` y `eg2ocr` (el nombre antiguo).

## Usar

```bash
amanu documento.pdf                           # Markdown
amanu pagina.jpg -f tei -o salida/            # salida/pagina.tei.xml
amanu carpeta/ -f md,page,alto,json -o x/     # lote: todos los PDF e imágenes, varios formatos
amanu escaneos/ -r -o x/                      # rápido: solo los lectores diminutos (~0,1-0,2 s por página)
amanu folio.png -m historico                  # el lector grande para manuscritos
amanu libro.pdf --modo ocr --paginas 10       # ignorar la capa de texto; solo 10 páginas
amanu escucha entrevista.mp3 -f srt,txt -o subtitulos/
amanu modelos                                 # catálogo y qué hay descargado
amanu visor                                   # visor local en http://127.0.0.1:8766/
```

Lectores (`-m`): `auto` (por defecto: Ligero, y las líneas dudosas se releen con el histórico y el multilingüe),
`ligero`, `minimo`, `historico`, `multilingue`, `impreso`.

Desde Python:

```python
import amanuensis
doc = amanuensis.leer("documento.pdf")
print(doc.exportar("md"))            # "txt", "json", "tei", "page", "alto", "hocr"
voz = amanuensis.escuchar("audio.mp3")
print(voz["texto"], voz["palabras"][0])  # {"texto", "inicio", "fin", "conf"}
```

## Modelos

| nombre | qué es | tamaño | licencia de uso |
|---|---|---|---|
| `ligero` | Amanuensis Ligero, lector de CPU (3,5 M) | 14 MB | no comercial* |
| `minimo` | Amanuensis Mínimo, lector de CPU (1,07 M) | 4 MB | no comercial* |
| `detector-minimo` | detector de líneas de CPU (0,13 M) | 0,5 MB | Apache 2.0 |
| `historico` | EmbeddingGemma 2 + LoRA, manuscritos e impresos s. XII-XX | 186 MB | no comercial* |
| `multilingue` | EmbeddingGemma 2 + LoRA, ≈14 800 caracteres | 193 MB | no comercial* |
| `voz` | voz a texto, 5 lenguas, tiempos por palabra | 351 MB | Apache 2.0 |

\* Parte de los datos de entrenamiento (HSMS, OHG) es CC BY-NC 4.0. Detalles, datos y licencias de origen en
[MODEL_CARD.md](MODEL_CARD.md).

## Límites conocidos

- La segmentación de páginas de manuscrito con el detector diminuto es débil (está entrenado con impreso): para
  manuscritos rinde mucho mejor sobre imágenes de línea. La maquetación completa (zonas SegmOnto, líneas base,
  orden aprendido) llega en la próxima versión.
- La transcripción de manuscritos es grafemática: no desarrolla abreviaturas.
- En CPU, el lector grande tarda de 2 a 10 s por página; los diminutos, décimas.

## Variables de entorno

`AMANUENSIS_CACHE` (pesos; por defecto `~/.cache/amanuensis`), `AMANUENSIS_ORIGEN` (otro servidor de pesos),
`AMANUENSIS_MODELOS` (carpeta local con los modelos), `AMANUENSIS_VOZ` (carpeta de un modelo de voz).

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
amanu scans/ -r -o out/                      # fast mode: tiny CPU readers only
amanu escucha interview.mp3 -f srt -o subs/  # speech to subtitles with word timings
```

- **Tiny and accurate**: Amanuensis Mínimo (1.07 M parameters) matches Indra OCR's speed on a laptop CPU
  (34 pages/s with 14 processes) with a third of its character errors on modern print lines (0.28 % vs 0.84 %),
  half on degraded scans (4.46 % vs 9.06 %), and 28 % vs 76 % on medieval and early modern manuscripts.
- **Digital PDFs** use their own text layer when it passes sanity checks; only the missing pages are OCRed.
- **Outputs**: Markdown, plain text, JSON, TEI, PAGE XML, ALTO, hOCR; SRT, VTT and JSON for speech.
- **Local and private**: no cloud, no GPU needed. Weights download on first use.
- **Licences**: code Apache 2.0. Some weights are non-commercial because part of their training data is
  CC BY-NC 4.0; see [MODEL_CARD.md](MODEL_CARD.md).

Author: José Luis Saorín Ferrer. Demo in the browser: https://lector-eg2.pages.dev
