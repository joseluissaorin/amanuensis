# Cambios

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
