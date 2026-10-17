# Amanuensis: ficha de los modelos / model card

Autor: José Luis Saorín Ferrer. Versión de los pesos: octubre de 2026 (familia v3 temprana).
Código: Apache 2.0. Pesos: ver la licencia de uso de cada modelo; donde parte de los datos de entrenamiento es
CC BY-NC, el modelo se publica para **uso no comercial**.

Todos los modelos de visión «grandes» parten del codificador de visión de **EmbeddingGemma 2** (Google, Apache 2.0),
congelado y recortado a 14 capas, con adaptadores LoRA y cabezas CTC propios. Los modelos de voz parten de su
codificador de audio. Ninguno contiene ni ha usado salidas ni pesos de Indra OCR ni de otros sistemas de OCR de
terceros. Ningún modelo genera texto libre: son lectores CTC (cada carácter sale de una posición de la imagen).

Bancos de medida: propios, sin solape de documentos con el entrenamiento (comprobado por documento). CER = tasa de
error de caracteres en %.

## Amanuensis Ligero (`ligero`, 3,5 M parámetros) y Amanuensis Mínimo (`minimo`, 1,07 M)

- Arquitectura propia: tallo convolucional 2D, bloques 1D con convolución por canal y excitación, dos capas de
  atención; CTC sobre 617 clases (alfabeto latino moderno e histórico con MUFI). Entrada: línea a 40 px de alto.
  ONNX fp32 (14 MB y 4 MB).
- Entrenamiento: alumnos del lector grande (misma mezcla de datos, mismas convenciones), 80 000 pasos de 256 líneas.
- Medidas: impreso moderno, líneas limpias / degradadas: Ligero 0,21 / 4,00, Mínimo 0,28 / 4,46 (Indra OCR 0,84 /
  9,06 en el mismo banco). Manuscritos e impresos antiguos (1214 líneas): Ligero 24,9, Mínimo 28,1 (Indra 76,1).
- **Licencia de uso: no comercial** (por los datos CC BY-NC de la tabla).

## Detector Mínimo (`detector-minimo`, 0,13 M)

- U-Net de bloques invertidos; mapas de núcleo y región de línea a paso 4. F1 de cajas 0,92 a 768 px en validación.
- Datos: páginas de PDF nacidos digitales (corpus privado de documentos, no se redistribuye) con sus cajas de línea,
  con aumentos de escaneo, giro, curvatura y sombra. Entrenado solo con impreso: en manuscritos segmenta mal.
- Licencia de uso: Apache 2.0.

## Histórico (`historico`, v2b96) y Multilingüe (`multilingue`, ml_v1)

- EmbeddingGemma 2 (visión, 14 capas) + LoRA de rango 32 + cabeza CTC; 8 bits por fila (ONNX MatMulNBits).
- Histórico: líneas a 96 px; manuscritos e impresos de los siglos XII-XX; banco de manuscritos CER 20,0; impreso
  moderno 0,42 / 4,27.
- Multilingüe: líneas a 64 px, ≈14 800 caracteres de todas las escrituras; parte del histórico y añade líneas de PDF
  multilingües y sintéticas (Wikipedia, CC BY-SA, con tipos Noto, OFL).
- **Licencia de uso: no comercial** (datos CC BY-NC de la tabla).

## Impreso (`impreso`, r2) y Detector (`detector`, det2)

- EmbeddingGemma 2 + cabeza CTC (sin LoRA), líneas a 32 px; detector sobre las capas 6, 10 y 14 del mismo codificador.
- Datos: corpus privado de PDF nacidos digitales (no se redistribuye) y líneas sintéticas con tipos de licencia libre.
- Licencia de uso: Apache 2.0.

## Voz (`voz`, stt-h)

- Codificador de audio de EmbeddingGemma 2 con LoRA (r 16) + cabeza CTC (transformador de 6 bloques); salidas de
  texto con puntuación y mayúsculas, idioma y tiempos por palabra (40 ms). ONNX con cuantización dinámica de 8 bits.
- Lenguas: español, catalán, gallego, euskera y portugués. FLEURS con modelo de lengua (WER): es 7,7, ca 12,0,
  gl 13,2, pt 12,4; Common Voice eu 12,7 (Whisper large-v3-turbo: 2,8 / 5,8 / 12,7 / 4,4 / 41,5).
- Datos: Multilingual LibriSpeech (CC BY 4.0), Common Voice 17 (CC0), VoxPopuli (CC0), FLEURS train (CC BY 4.0).
- Licencia de uso: Apache 2.0.

## Datos de entrenamiento de los lectores de texto y sus licencias

| conjunto | contenido | licencia de origen |
|---|---|---|
| CATMuS Medieval 1.6 | manuscritos s. VIII-XVI, latín y romances | CC BY 4.0 |
| CATMuS Modern 1.0 | manuscritos e impresos s. XV-XX | CC BY 4.0 |
| TRIDIS v1 (HOME-Alcar, Himanis, CODEA, e-NDP…) | documentos s. XII-XVII | MIT (fuentes CC BY) |
| iForal | forais portugueses 1217-1491 | CC BY 4.0 |
| **Spanish Gothic Incunabula (HSMS)** | incunables castellanos 1472-1500 | **CC BY-NC 4.0** |
| HTR-United y otros: Caroline, chapbooks, FoNDUE, Liber, NUBIS, Paris Bible, Dresdner Hofdiarium, Finnish court records, Inzigkofen, NewsEye, Nuremberg letterbooks, ONB 3891, Paderov Bible, TranscriboQuest, graphemic ENHG | manuscritos e impresos s. IX-XX | CC BY 4.0 o CC BY-SA 4.0 |
| Reichenau (deu, lat) | manuscritos | CC BY-SA 4.0 |
| **OHG (Old High German)** | manuscritos | **CC BY-NC 4.0** |
| Wikisource (ca, de, en, es, fr, gl, it, la, pt) | facsímiles de dominio público con su transcripción | CC BY-SA 4.0 (texto) |
| Líneas de PDF nacidos digitales | impreso moderno | corpus privado; no se redistribuye |
| Sintéticas (moderno e histórico) | textos de Wikipedia y Gutenberg con tipos libres | CC BY-SA / dominio público; tipos OFL |
| Pseudoetiquetas de manuscritos | imágenes de Internet Archive etiquetadas por acuerdo entre un modelo de Gemini y un modelo de lengua | condiciones de cada facsímil |

Descriptores de imagen: Iconclass (CC0), 7071 notaciones de hasta cinco caracteres, recuperadas por similitud en el
espacio común imagen-texto de EmbeddingGemma 2; se ofrecen como sugerencias, nunca se genera texto.

Los conjuntos reservados para el banco de examen (entre otros, GT4HistOCR, Bullinger, Gwalther, Rodrigo, BNE
INC/901) no se usaron en ningún entrenamiento.

---

**English summary.** Code: Apache 2.0. Weights `ligero`, `minimo`, `historico` and `multilingue` are released for
**non-commercial use** because part of their training data (HSMS Spanish Gothic Incunabula and OHG) is
CC BY-NC 4.0. `detector-minimo`, `impreso`, `detector` and `voz` are Apache 2.0. All vision models build on
Google's EmbeddingGemma 2 (Apache 2.0); none uses weights or outputs of Indra OCR or other third-party OCR systems.
