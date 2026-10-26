# Amanuensis: ficha de los modelos / model card

Autor: José Luis Saorín Ferrer. Versión de los pesos: octubre de 2026 (Amanuensis 1.0; paquete 1.1).
Código: Apache 2.0. Pesos: ver la licencia de uso de cada modelo; donde parte de los datos de entrenamiento es
CC BY-NC, el modelo se publica para **uso no comercial**.

Todos los modelos de visión «grandes» parten del codificador de visión de **EmbeddingGemma 2** (Google, Apache 2.0),
congelado y recortado a 14 capas, con adaptadores LoRA y cabezas CTC propios. Los modelos de voz parten de su
codificador de audio. Ninguno contiene ni ha usado salidas ni pesos de Indra OCR ni de otros sistemas de OCR de
terceros. Ningún modelo genera texto libre: son lectores CTC (cada carácter sale de una posición de la imagen).

Bancos de medida: propios, sin solape de documentos con el entrenamiento (comprobado por documento). CER = tasa de
error de caracteres en %.

La familia 1.0 tiene tres miembros: **Amanuensis** (el modelo grande, la página entera), **Amanuensis Ligero** (CPU;
con `--rapido` usa por dentro el lector Mínimo) y **Amanuensis Voz** (módulo opcional). Con `-m auto` (por defecto)
cada página va a Amanuensis o a Ligero según lo que sea. Las demás piezas (detector Mínimo, orden ligero, lectores
histórico, multilingüe e impreso) son internas o para expertos y se describen al final.

## Amanuensis (`amanuensis`; antes «completo», que sigue valiendo como alias)

La página entera de una vez: renglones de alta resolución, maquetación (zonas SegmOnto y DocLayNet, tipos de línea,
tablas con filas y columnas, orden de lectura aprendido), recorte de cada línea enderezado por su línea base y lector
grande con selector por escritura (histórico para el alfabeto latino, multilingüe para las demás); las páginas con 200
renglones o más pasan por la rama densa, y el orden final se elige por continuidad del texto (modelo de n-gramas de
caracteres, `continuidad.npz`). La página se endereza antes por la geometría de sus renglones (ángulo libre y giros de
90, 180 y 270 grados) y por la orientación EXIF de las fotos. ONNX Runtime con
8 bits por pesos (MatMulNBits, bloques de 32), sin PyTorch; en el Mac, lector y maquetación en Core ML fp16 (GPU).
Con `-m auto` (por defecto) una sonda barata (detector Mínimo + Ligero) deja en Ligero el impreso moderno latino de
una columna y manda lo demás a Amanuensis.

| pieza | qué es | datos | licencia de uso |
|---|---|---|---|
| codificador base (`cod_base_q8`) | EmbeddingGemma 2 visión recortado a 14 capas, sin adaptador | Google | Apache 2.0 (términos de Gemma) |
| renglones LineasAR (`lineas_fp32`) | mapas de núcleo y línea base a paso 2 sobre las capas 6, 10 y 14 | páginas con línea base de colecciones abiertas y páginas sintéticas | Apache 2.0 (licencias de las 120 colecciones de origen por confirmar una a una) |
| maquetación v4 enfriada, mq_v4e (`cod_mq_q8`, `maqueta_fp32`, `orden_fp32`, `orden_par`) | LoRA propio + cabezas de zonas, líneas, tablas y orden | XML de colecciones abiertas, consenso, DocLayNet (CDLA-Permissive 1.0), cBAD, sintéticas | **no comercial** mientras su LoRA parta del lector v3 (datos CC BY-NC) |
| rama densa, mq_v4o (`densa/cod_mq_q8`, `densa/maqueta_fp32`, `densa/orden_fp32`, `densa/orden_par`) | la maquetación v4 con la cabeza de orden reajustada para prensa (líneas detectadas, página entera); se usa en páginas con 200 renglones o más, con orden jerárquico (zonas y luego renglones) y renglones por teselas cuando la letra es muy pequeña | los de la maquetación más prensa real (NewsEye fr/fi CC BY, BnL CC0) y sintética densa | **no comercial** (parte del mismo LoRA) |
| lector grande (`lector/`) | l3_interp_r5_050: interpolación al 50 % del adaptador del lector v3 y su ronda 3 («v3 + r3, α 0,5»), interpolada a su vez al 50 % con la ronda 5 (EmbeddingGemma 2 + LoRA de rango 128 + cabeza CTC de 14 653 clases, líneas a 96 px) | los de la tabla de abajo | **no comercial** (parte del v3, entrenado con HSMS y OHG, CC BY-NC) |
| multilingüe (`multilingue`, se baja aparte) | ml_v1 | ver más arriba | **no comercial** |

El conjunto se distribuye para **uso no comercial** mientras alguna pieza lo sea.

- Paridad con PyTorch (13 páginas variadas, CER de página frente a la verdad): ONNX de 8 bits 34,98 y ONNX + Core ML
  34,99, frente a 43,8 de la versión anterior (lector v3 + maquetación v4) en PyTorch. Banco de 320 líneas: 6,11 en
  ONNX frente a 6,06 de PyTorch.
- Velocidad por página (Mac M4 Max con otros trabajos en marcha): Core ML 7,2 s, solo CPU 10,9 s; Xeon de 2016 sin GPU
  (versión anterior) 36,8 s; PyTorch en A100 0,7 s. En CPU sin GPU el lector grande es el 70-90 % del tiempo: para
  lotes grandes de impreso moderno, `-m ligero`.
- Lo que aún hace mal: prensa a muchas columnas (el orden intercala columnas; CER de página ~70 en el examen) y letra
  moderna. Ver el informe técnico.

## Amanuensis Ligero (`ligero`, 3,5 M parámetros; modo rápido con el lector Mínimo, 1,07 M)

- Página: maquetación mínima (`maqueta-minima`, 2,5 M: renglones con línea base curva, zonas, orden y orientación;
  recorte enderezado por la línea base) y orden por continuidad del texto. En el modo automático, la sonda sigue usando
  el detector Mínimo y su orden ligero para el impreso moderno limpio. `AMANUENSIS_MAQUETA_MINIMA=0` vuelve al detector.

- Arquitectura propia: tallo convolucional 2D, bloques 1D con convolución por canal y excitación, dos capas de
  atención; CTC sobre 617 clases (alfabeto latino moderno e histórico con MUFI). Entrada: línea a 40 px de alto.
  ONNX fp32 (14 MB y 4 MB).
- Entrenamiento: alumnos del lector grande (misma mezcla de datos, mismas convenciones), 80 000 pasos de 256 líneas.
- Versión 2 (`amanuensis-ligero-2`, `amanuensis-minimo-2`): re-destiladas del lector grande v3 (pseudoetiquetas
  del v3 sobre CoMMA y oro de «lector3»), 30 000 pasos con guardia de impreso moderno.
- Medidas (versión 2; entre paréntesis, la 1): impreso moderno, líneas limpias / degradadas: Ligero 0,20 / 3,88
  (0,21 / 4,00), Mínimo 0,26 / 4,40 (0,28 / 4,46); Indra OCR 0,84 / 9,06 en el mismo banco. Manuscritos e impresos
  antiguos (1214 líneas): Ligero 23,8 (24,9), Mínimo 27,4 (28,1); Indra 76,1.
- Versión 3 (`amanuensis-ligero-3`, `amanuensis-minimo-3`, la de la 1.0): re-destiladas del lector l3_v3r2 con impreso
  «de la vida real» (fotos de móvil, fotocopias, fax, microfilm, baja resolución), mecanografiado sintético y facsímiles
  escaneados; guardia de impreso moderno, manuscritos, mecanografiado, baja resolución y foto.
- Examen sellado (líneas, CER %; Ligero 3 / Ligero 2; Mínimo 3 / Mínimo 2): media de 8 grupos 21,5 / 22,2; 24,2 / 25,0;
  impreso moderno 6,67 / 8,56; 8,45 / 11,0; degradada 10,2 / 15,1; 12,2 / 17,2 (lector grande: 15,3 de media, 5,06 en
  impreso moderno). Páginas de punta a punta: solo sirven para impreso latino de una columna (Wikisource 25,5).
- **Licencia de uso: no comercial** (por los datos CC BY-NC de la tabla).

## Amanuensis Voz (`voz`, voz2; módulo opcional) y su modelo de lengua (`voz-lm`)

- Codificador de audio de EmbeddingGemma 2 (12 capas) con LoRA de rango 32 fundido + cabeza CTC propia (8 bloques,
  44 M) con salida plana y salida rica (mayúsculas y puntuación) de 16 384 piezas, cabeza de idioma y una ficha cada
  40 ms. 40 lenguas, unas 21 000 h de audio. ONNX de 8 bits con calibración estática (481 MB): WER con LM 10,13
  frente a 10,12 de PyTorch en el banco de comprobación.
- FLEURS prueba con LM (WER): es 5,88, ca 8,69, gl 11,08, pt 9,79; Common Voice eu 8,96, gl 6,62 (Whisper
  large-v3-turbo: 2,79 / 5,33 / 12,79 / 4,38; 40,95 / 16,43). En las lenguas grandes Whisper y Parakeet son
  claramente mejores (media de 27 lenguas 19,7 frente a 12,3).
- Datos: YODAS-Granary (CC BY 3.0), Common Voice 17 (CC0), Multilingual LibriSpeech (CC BY 4.0), VoxPopuli (CC0),
  FLEURS entrenamiento (CC BY 4.0), YODAS2 manual (CC BY 3.0), HiTZ euskera (CC BY 4.0). Pesos: Apache 2.0.
- Modelos de lengua KenLM de 4-gramas, 37 lenguas, bajo demanda: Common Voice (CC0) y Wikipedia (CC BY-SA 4.0); se
  distribuyen con CC BY-SA 4.0.

### Voz anterior (`voz-h`, stt-h, versión 0.4): fallo conocido

El ONNX publicado de stt-h (`amanuensis-voz-h-1`) tiene un fallo de exportación en la máscara de atención del
codificador: cada ficha ve una ficha de más a la izquierda (`q - kv <= 12` en lugar de `< 12`). Frente a PyTorch,
la WER entre ambas salidas es del 3,4 %. No se vuelve a exportar: voz2 lo sustituye en cuanto se publique, y
mientras tanto `voz` cae en `voz-h` solo si voz2 no está disponible. Lenguas de stt-h: es, ca, gl, eu, pt.
Licencia: Apache 2.0.

## Piezas internas y lectores para expertos

No se presentan como modelos aparte: el detector Mínimo y el orden ligero los usa Ligero; el histórico y el
multilingüe, Amanuensis (y `-m historico` / `-m multilingue` para expertos); el impreso y su detector son de la
versión 0.4. La maquetación mínima está integrada pero apagada (`AMANUENSIS_MAQUETA_MINIMA=1`).

### Detector Mínimo (`detector-minimo`, 0,13 M)

- U-Net de bloques invertidos; mapas de núcleo y región de línea a paso 4. F1 de cajas 0,92 a 768 px en validación.
- Datos: páginas de PDF nacidos digitales (corpus privado de documentos, no se redistribuye) con sus cajas de línea,
  con aumentos de escaneo, giro, curvatura y sombra. Entrenado solo con impreso: en manuscritos segmenta mal.
- Licencia de uso: Apache 2.0.

### Orden de lectura ligero (empaquetado, `orden_ligero.npz`, 27 KB)

- Perceptrón de 3 capas (64) sobre 28 rasgos geométricos por pareja de líneas, con contexto de página; decodificación
  por inserciones. Entrenado con las cajas que da el detector Mínimo en 12 067 páginas de PDF nacidos digitales
  (corpus privado, documentos disjuntos de los bancos), con el orden de sus líneas en el PDF como referencia.
  Acierto de pares en validación (665 páginas): 95,7 %. Licencia de uso: Apache 2.0.

### Orden de lectura por grafo (empaquetado, `orden_gcn.npz` y `orden_gcn_giro.npz`, 235 KB cada uno)

- Red de grafo disperso sobre las cajas de renglón (vecinos más próximos y vecinos por dirección; 4 capas de paso de
  mensajes condicionado por la arista, d = 64), con una cabeza de sucesor y otra de precedencia por parejas;
  decodificación en cadenas y orden de las cadenas por inserciones. Inferencia en numpy, pesos en fp16. Se usa por
  defecto en las páginas a dos columnas o más; los dos modelos van en pareja (2/3 y 1/3; el segundo, entrenado con
  giros de hasta ±2°, aguanta la inclinación residual).
- Solo ve la geometría de las cajas (posición y tamaño), nunca la imagen ni el texto. Entrenado en CPU con el orden de
  páginas de PDF nacidos digitales (corpus privado, el del orden ligero), prensa con orden de lectura anotado y páginas
  sintéticas de libro, revista, estructura y fotos de móvil.
- Validación (pares en orden · páginas perfectas): 97,3 % · 61 % en conjunto, frente a 93,6 % · 26 % del orden ligero.
- Pesos propios de José Luis Saorín. **Licencia de uso: Apache 2.0.**

### Continuidad del texto (empaquetado, `continuidad.npz`, 2,6 MB)

- Modelo de lengua de caracteres diminuto (n-gramas de hasta 5 caracteres con Witten-Bell, tablas con dispersión por
  función resumen y log-probabilidades de un byte) que puntúa si un renglón sigue a otro; sirve para elegir entre
  órdenes de lectura candidatos. No contiene texto legible: solo recuentos dispersos.
- Datos: párrafos de la Wikipedia de 28 lenguas (volcado 20231101 de wikimedia/wikipedia; texto CC BY-SA 4.0) y
  transcripciones de manuscritos e impresos antiguos de entrenamiento (CATMuS Medieval y Modern, TRIDIS, iForal,
  Esposalles, AMSMB y **Spanish Gothic Incunabula (HSMS), CC BY-NC 4.0**).
- **Licencia de uso: no comercial** (por HSMS, CC BY-NC 4.0), con la atribución y la condición de compartir igual
  de la Wikipedia (CC BY-SA 4.0). Quien necesite un uso comercial puede borrar el fichero del paquete instalado: el
  motor vuelve a los órdenes sin continuidad.

### Perfiles de lengua (empaquetado, `perfiles.json`, 40 KB)

- Las 120 palabras funcionales más frecuentes y el alfabeto de cada una de las 31 lenguas de la corrección léxica,
  para detectar la lengua de la página sin red. Derivado de las listas de frecuencias de FrequencyWords 2018
  (Hermit Dave, subtítulos de OpenSubtitles; contenido CC BY-SA 4.0) y wikipedia-word-frequency (Ilya Semenov,
  texto de la Wikipedia, CC BY-SA 4.0). **Licencia: CC BY-SA 4.0.** Los léxicos completos no van en el paquete: se
  construyen en la máquina de quien lo usa (`amanu lexicos construir`) a partir de esas mismas fuentes.

### Histórico (`historico`, v2b96) y Multilingüe (`multilingue`, ml_v1)

- EmbeddingGemma 2 (visión, 14 capas) + LoRA de rango 32 + cabeza CTC; 8 bits por fila (ONNX MatMulNBits).
- Histórico: líneas a 96 px; manuscritos e impresos de los siglos XII-XX; banco de manuscritos CER 20,0; impreso
  moderno 0,42 / 4,27.
- Multilingüe: líneas a 64 px, ≈14 800 caracteres de todas las escrituras; parte del histórico y añade líneas de PDF
  multilingües y sintéticas (Wikipedia, CC BY-SA, con tipos Noto, OFL).
- **Licencia de uso: no comercial** (datos CC BY-NC de la tabla).

### Impreso (`impreso`, r2) y Detector (`detector`, det2)

- EmbeddingGemma 2 + cabeza CTC (sin LoRA), líneas a 32 px; detector sobre las capas 6, 10 y 14 del mismo codificador.
- Datos: corpus privado de PDF nacidos digitales (no se redistribuye) y líneas sintéticas con tipos de licencia libre.
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
| **Oficio de Hipotecas de Girona (OHG)** ([Zenodo 1322666](https://zenodo.org/records/1322666)) | escrituras notariales en español, s. XVIII | **CC BY-NC 4.0** |
| Wikisource (ca, de, en, es, fr, gl, it, la, pt) | facsímiles de dominio público con su transcripción | CC BY-SA 4.0 (texto) |
| Líneas de PDF nacidos digitales | impreso moderno | corpus privado; no se redistribuye |
| Sintéticas (moderno e histórico) | textos de Wikipedia y Gutenberg con tipos libres | CC BY-SA / dominio público; tipos OFL |
| Pseudoetiquetas de manuscritos | imágenes de Internet Archive etiquetadas por acuerdo entre un modelo de Gemini y un modelo de lengua | condiciones de cada facsímil |

Descriptores de imagen: Iconclass (CC0), 7071 notaciones de hasta cinco caracteres, recuperadas por similitud en el
espacio común imagen-texto de EmbeddingGemma 2; se ofrecen como sugerencias, nunca se genera texto.

Los conjuntos reservados para el banco de examen (entre otros, GT4HistOCR, Bullinger, Gwalther, Rodrigo, BNE
INC/901) no se usaron en ningún entrenamiento.

---

**English summary.** Amanuensis 1.0 is a family of three: **Amanuensis** (the large model: whole-page reading with
baselines, layout zones, tables, learned reading order and a large CTC reader with a script selector), **Amanuensis
Ligero** (3.5 M parameters, CPU; `--rapido` uses the 1.07 M Mínimo reader inside) and **Amanuensis Voz** (optional
speech-to-text module, 40 languages; clearly behind Whisper and Parakeet in the big languages, ahead in Galician,
Basque and Catalan Common Voice). `-m auto` sends each page to Amanuensis or Ligero. Code: Apache 2.0. Amanuensis and
Ligero weights are released for **non-commercial use** because part of their training data (HSMS Spanish Gothic
Incunabula and OHG, the Oficio de Hipotecas de Girona collection of 18th-century Spanish notarial deeds) is CC BY-NC 4.0. Voz weights are Apache 2.0; its language models are CC BY-SA 4.0. The previously
published `voz-h` (stt-h) ONNX has an attention-mask export bug (one extra frame of left context; 3.4 % WER between
ONNX and PyTorch); voz2 replaces it. All vision models build on Google's EmbeddingGemma 2; none uses weights or outputs
of Indra OCR or other third-party OCR systems. The sparse-graph reading-order weights (`orden_gcn.npz`, `orden_gcn_giro.npz`), like the Mínimo detector and
the light reading order, are Apache 2.0. The bundled text-continuity model (`continuidad.npz`, character n-grams)
is **non-commercial** (part of its training text is HSMS, CC BY-NC 4.0) and carries the CC BY-SA 4.0 attribution of
Wikipedia; the bundled language profiles (`perfiles.json`) are CC BY-SA 4.0. See NOTICE. The optional word lists for lexical correction derive from OpenSubtitles
(FrequencyWords) and Wikipedia word frequencies, both CC BY-SA 4.0.
