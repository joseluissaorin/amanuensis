# Cambios

## 1.2.0 (11-11-2026): prensa, fotos y velocidad

Campaña «prensa y fotos» sobre el corpus de prensa histórica GRESEL (UNED). Todo lo que sigue es opcional o se activa solo con
`--prensa si`: el impreso moderno, los manuscritos y los demás modos no cambian.

- **Modo prensa afinado** (`--prensa si`): lector de prensa (R3), maquetación y orden de lectura v5p, detector de renglones i50 y
  haz con modelo de lengua de caracteres (alfa 0,3, beta 1, delta 8, sin tocar tildes ni mayúsculas). Prueba de GRESEL, 88 págs.,
  CER de página 15,95 (1.1.1 sin modo prensa) → 9,20 (primer modo prensa) → **8,00**; Ligero con `--prensa si`, **8,44**
  (`-m ligero --prensa si` usa la maquetación del modelo completo y el lector de Ligero). Con `--prensa auto`, una página que
  Ligero lee como prensa se relee con ese modo.
- **Fotos**: `--aplanar auto|si|no` (UVDoc, MIT, con selector «solo si hace falta»; solo imágenes, nunca PDF). Banco de 16 fotos reales
  (referencia plata): CER 33,3 → **26,4**. En escaneos planos el selector salta alguna vez (1 de 90 en validación): por defecto no.
- **Artículos**: `paginas[].articulos` en el JSON, separador en Markdown y `<div type="article">` en TEI; reglas por defecto y
  `--articulos red` (cabeza aprendida, B-cubed **88,1** frente a 88,2 de dots.ocr y 91,2 de Chandra 2 sobre la misma verdad *plata*
  de Gemini, no humana; 79,8 las reglas). Necesita `pip install "amanuensis[articulos] @ git+https://github.com/joseluissaorin/amanuensis"` y ~960 MB de pesos.
- **Velocidad**: Core ML del codificador de página y del lector de prensa en el Mac (−27 % de tiempo por página); en GPU de NVIDIA
  con CUDA 12, hasta ~1 s por página con varios procesos (L40S, 0,97 págs./s frente a 0,28 de Chandra 2 y 0,17 de dots.ocr;
  ~0,23 $ por 1000 páginas). `--procesos N` reparte los ficheros. Un lote largo se relanza solo al pasar de ~3 GB para no
  rebasar los 6 GB.
- **Opcionales**: `--lector dots --servidor URL` (dots.ocr por un servidor de API OpenAI) y `--lector auto` (solo las páginas muy dudosas).
- **Catálogo** con sumas sha256 (se comprueban al bajar) y el modelo completo en una carpeta nueva (`amanuensis-2`); la de la 1.1
  sigue donde estaba. Piezas nuevas: lector y maquetación de prensa, `prensa-lm`, `prensa-lm-ligero`, `lm-neuronal`, `articulos-red`, `uvdoc`.
- **Ligero y Mínimo**: el rectificado y el pulido pasan a `auto` (apagados en Ligero y Mínimo, donde empeoraban el impreso moderno y
  Wikisource); el haz con modelo de lengua es de serie en Ligero con `--prensa si`.
- Limitaciones: las cifras de prensa salen del propio corpus de GRESEL (la prueba no entró en ningún entrenamiento, pero es la misma
  hemeroteca); La Vanguardia de Manila a ~100 ppp (3 planas no vistas) sigue en CER ~15.

## Modo prensa, primera versión (10-11-2026, dentro de la 1.2.0)

Prensa a varias columnas (`--prensa auto|si|no`, `Motor(prensa=…)`, `AMANUENSIS_PRENSA`; `auto` solo actúa con tres
columnas o más, `AMANUENSIS_PRENSA_COLUMNAS` cambia el umbral). Nuevo módulo `prensa.py`; actúa después del modelo
completo y de la sonda de Ligero, sin pesos nuevos.

- **Renglones dobles**: una caja con dos o más bandas de tinta separadas por un valle se parte por el perfil y cada trozo
  se relee con el lector grande; el corte se acepta si los trozos leen al menos 1,4 veces más letras que la caja entera,
  y los renglones repetidos se quitan. Las franjas del borde se recuperan solo si no repiten un renglón vecino.
- **Cortes por el canal** con la x de cada carácter: el canal se deduce de los renglones vecinos (acaban a un lado y
  empiezan al otro; casi ninguno lo cruza).
- **Orden de Breuel (2003)** con solape relativo al ancho (las cajas invaden unos píxeles la columna vecina) y los trozos
  de un mismo titular agrupados (misma zona, letra de titular, mismo estilo de mayúsculas). El corte XY por filas
  intercalaba columnas cuando los renglones se alinean de una a otra.
- **Bloques de columna** como zonas: TXT, Markdown, TEI y PAGE XML salen por columnas y artículos, con orden de lectura.
- Medido con el corpus de GRESEL (repo greseluned/olmOCR-finetuning, PAGE de Transkribus; sin las páginas cuya
  transcripción es de otra página): prueba, 88 págs., CER de página 15,95 → **9,20** (mediana 11,60 → 5,37), CER por región
  10,40 → **6,31**, WER 20,94 → 13,59 (Tesseract 5 spa: 17,25 de CER y 29,73 de WER); validación, 90 págs., 14,04 → **9,60**
  y 7,57 → **4,72**; La Vanguardia de Manila, 9 págs., 54,54 → **24,24** y 38,05 → **8,55**. Mejoran 47 páginas y empeoran 13
  (prueba). Límite conocido: los sumarios con título y autor en la misma fila se leen columna a columna.
- Coste: unos segundos más por plana (relectura de los renglones dobles).

## 1.1.1 (26-10-2026)

La maquetación de Amanuensis Ligero y de su modo rápido se elige **página a página**.

- **Regla por página** (`AMANUENSIS_MAQUETA_MINIMA=pagina`, nuevo defecto): la página se lee primero con la maquetación
  clásica (detector Mínimo, orientación, columnas, orden por grafo en prensa y revistas). Si más del 10 % de sus renglones
  con texto salen con confianza < 0,75, o hay menos de 3, se vuelve a leer con la maquetación mínima (renglones curvos,
  zonas, orden por continuidad). La 1.1.0 usaba siempre la mínima, que gana en manuscritos, impreso antiguo, prensa y
  fotos, pero pierde en impreso moderno: repite renglones, deja folios y cabeceras y parte párrafos por zonas, y con ella
  no actuaban las columnas ni la palabra partida de la 1.1. `AMANUENSIS_MAQUETA_MINIMA=auto` vuelve a lo de la 1.1.0 y `0`
  usa siempre la clásica.
- Medido en páginas de desarrollo (324, sin el examen sellado; FCA / CER medios por página, %): Ligero 33,8 / 39,2
  (1.1.0) → 34,5 / **38,0**; modo rápido 36,8 / 41,2 → **36,2 / 39,2**. Por tipo de página, con Ligero: impreso moderno a
  varias columnas 10,4 → 6,5 de FCA, prensa 31,5 → 26,4 de CER, y los manuscritos siguen igual (43,0 → 43,5 de FCA). En
  los bancos de orden (WER del texto de lectura) la regla da lo de la clásica: Grotowski 19,80 → 6,15, Implicaturas
  3,07 → 0,78, Pragmática 3,24 → 0,23, La comunicación 6,09 → 1,89, Anchieta 10,64 → 4,38, Grice 2,92 → 0,49. En 34 páginas
  inclinadas, 18,71 → 4,07; sin folios sueltos.
- **Modo rápido**: la maquetación mínima ya no gira la página con su clasificador de orientación cuando no se endereza.
  Giraba 90° o 180° páginas que estaban derechas: 18 de 324 en desarrollo.
- Coste: en Ligero, las páginas que no son de impreso moderno se leen dos veces (de 1,1 a 1,5 s por página de media en
  CPU). En el modo rápido apenas cambia.

## 1.1.0 (26-10-2026)

Lo que va de leer a entregar un texto que se pueda leer, sin tocar los modelos.

- **Texto de lectura por defecto** en Markdown, TXT y HTML: sin folios, cabeceras, signaturas, marcas de página ni
  restos del escaneo; párrafos y palabras partidas unidos también entre páginas y entre las dos hojas de una doble
  página; el verso en sus renglones; notas al final. `--fiel` (y `Documento.fiel`) da lo de antes.
- **Dobles páginas** partidas por el canal y leídas hoja a hoja (`doble.py`).
- **Renglones curvos** enderezados tira a tira y releídos (`rectificado.py`, `--rectificar`).
- **Pulido de la página** (`pulido.py`, `--pulido`): estructura de las páginas sin maquetación (cabeceras, folios,
  epígrafes con su nivel, títulos de capítulo, figuras, filas partidas, basura) y series en todas (romanos y letras de
  las enumeraciones, ejemplos, llamadas de nota, números de verso y de epígrafe, personajes); diacríticos imposibles en
  las lenguas del documento.
- **Corrección léxica guiada por la CTC en 31 lenguas** (`lexico.py`, `--lexico`): detección de lengua por página
  (`perfiles.json` empaquetado), léxicos construidos la primera vez que hacen falta (`amanu lexicos`), confusiones
  sistemáticas aprendidas de cada documento, autoléxico del propio documento, guardas para textos antiguos y lenguas
  sin léxico.
- **Signos** (`signos.py` y la decodificación del lector): comillas tímidas (el lector las ve con un 10-45 % y el
  blanco les ganaba) y comillas por parejas; raya media o larga por su medida en la imagen, con los anchos del
  documento; ⊢ por su forma; notación propia del documento («+>», «>>»). Series más robustas: llamadas de nota tras
  un signo («1996)3»), apartados largos, ejemplos con un paréntesis perdido («4)», «(8») y remisiones que no
  descuadran la serie.
- **Columnas de prensa y revista** (`columnas.py`): comienzos de columna alineados, trozos de renglón unidos (con tinta
  entre medias), cajas partidas por el canal, orden por columnas y por la palabra partida que une dos bloques. En una
  revista de teatro a tres columnas con fotos y lápiz (Máscara, Grotowski), WER frente a Gemini 22,7 % → 12,7 %.
- `Motor(pulido=..., lexico=..., rectificar=...)` y `amanuensis.leer(..., fiel=...)` para usarlo desde una app.
- **Orden de lectura por grafo disperso** (`orden_gcn.py`, GCN sobre las cajas de renglón) por defecto en las páginas a
  dos columnas o más, sobre las cajas ya unidas y partidas por el canal: Máscara (revista de teatro a tres columnas) 12,6 →
  6,2 % de WER frente a Gemini (página 2: 52,8 → 13,5 %); los libros a una columna, igual. Con `AMANUENSIS_ORDEN=gcn` en
  todas las páginas y tamaños (detector, maqueta mínima y modelo grande): en Anchieta (teatro con los personajes al
  margen) 4,32 → 4,87 %, por eso no es el defecto. `ligero` y `columnas` fuerzan los órdenes de antes. Sin pesos del grafo,
  el orden de siempre. *Sparse-graph reading order by default on multi-column pages.*
- DocLayout-YOLO solo se usa si se pide (`AMANUENSIS_MAQUETA_YOLO_SI=1`, o `AMANUENSIS_MAQUETA_YOLO` con la carpeta):
  tenerlo en la caché ya no cambia la lectura, que así es la misma que la de la demo, Yappy y la app. Sin él, Grotowski
  queda en 6,15 % de WER (6,09 con él); los otros cinco bancos, igual.
- DocLayout-YOLO solo clasifica en páginas a dos columnas o más: en libros a una columna tomaba folios, títulos y notas
  por figuras o pies (Grice 0,56 → 5,35 % de WER, Pragmática 0,19 → 2,21 %).
- El orden por la palabra partida ya no se aplica a las dobles páginas (las dos hojas no son dos columnas: La
  comunicación 2,31 → 6,11 % de WER); `AMANUENSIS_CONTINUIDAD_BLOQUES=0` lo apaga.
- `bench/orden_bancos.py`: los seis bancos del orden con Ligero (datos en `datos/bancos_orden`).
- Ficheros de licencia: NOTICE nuevo; `continuidad.npz`, uso no comercial (HSMS, CC BY-NC 4.0) y atribución
  CC BY-SA de la Wikipedia; `perfiles.json`, CC BY-SA 4.0.
- Catálogo: Amanuensis, Ligero 3, Mínimo 3, la maquetación mínima, Voz 2 y sus modelos de lengua, publicados en el
  almacén de modelos (antes, pendientes).

Capítulo nuevo («Las implicaturas conversacionales», 7.163 palabras, referencia de Gemini Pro arbitrada contra el
escaneo): 111 → 46 palabras mal (1,55 % → 0,64 %); los signos eran más de la mitad.

Medido contra Gemini Pro: impreso moderno limpio, WER 3,17 % → 0,19 % (Gemini Flash 0,21 %); fotocopia curvada de un
libro, 6,65 % → 2,37 % (Flash 0,41 %); teatro de Anchieta en tupí, portugués y español, 7,99 % → 4,90 % (Flash 1,0 %).

## 1.0.0 (19-10-2026)

La familia: Amanuensis, Amanuensis Ligero y Amanuensis Voz.
