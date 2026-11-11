"""Rivales VLM sobre las páginas de prensa de GRESEL (datos/gresel/{test,lv}) contra un servidor vLLM local,
cada uno con su prompt, su tamaño de imagen y sus parámetros oficiales:

- olmocr   (allenai/olmOCR-2-7B-1025): prompt v4 YAML del paquete olmocr, PNG a 1288 px por el lado largo,
            temperatura 0,1, 8000 fichas; reintento a 0,2 si se corta (como su pipeline).
- chandra  (datalab-to/chandra-ocr-2): prompt «ocr_layout» y scale_to_fit del paquete chandra, temperatura 0,
            top_p 0,1, 12384 fichas, reintentos si repite; salida HTML -> Markdown con chandra.output.
- dots     (rednote-hilab/dots.ocr): prompt «prompt_layout_all_en» (JSON con cajas, categoría y texto en orden de
            lectura), temperatura 0,1, top_p 0,9; se une el texto de todo lo que no es «Picture».

Salida: resultados/gresel/<nombre>/<parte>/<stem>.txt (texto limpio) y .raw (salida tal cual).
python examen/gresel_cliente.py --modelo olmocr --api http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import html
import io
import json
import os
import re
import time

from openai import OpenAI
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
RAIZ = os.getcwd()  # se lanza desde la raíz del repositorio (/workspace/eg2-ocr en Vast)

OLMOCR = ("Attached is one page of a document that you must process. "
          "Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.\n"
          "If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
          "Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters.")
DOTS = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


def b64png(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def lado_largo(im, lado):
    z = lado / max(im.size)
    return im.resize((max(1, round(im.size[0] * z)), max(1, round(im.size[1] * z))), Image.LANCZOS)


def dots_fit(im, min_pixels=3136, max_pixels=11289600, factor=28):
    w, h = im.size
    s = 1.0
    if w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
    elif w * h < min_pixels:
        s = (min_pixels / (w * h)) ** 0.5
    nw, nh = max(factor, int(w * s // factor * factor)), max(factor, int(h * s // factor * factor))
    return im if (nw, nh) == (w, h) else im.resize((nw, nh), Image.LANCZOS)


def limpiar(t: str) -> str:
    t = re.sub(r"<br\s*/?>", "\n", t or "")
    t = re.sub(r"</(p|div|tr|li|h\d)>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"^```\w*|```$", "", t, flags=re.M)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"\*\*|__", "", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def sin_front_matter(t):
    m = re.match(r"^\s*---\s*\n.*?\n---\s*\n?", t or "", flags=re.S)
    return t[m.end():] if m else (t or "")


def repite(t, ventana=300):
    """Bucle de generación: el final se repite (lo que reintentan olmocr y chandra)."""
    if not t or len(t) < 2 * ventana:
        return False
    cola = t[-ventana:]
    return t[:-ventana].count(cola[-80:]) >= 3


def dots_texto(raw):
    try:
        el = json.loads(raw)
        if isinstance(el, dict):
            el = el.get("layout") or el.get("elements") or list(el.values())[0]
        partes = [e.get("text", "") for e in el if isinstance(e, dict) and e.get("category") != "Picture"]
    except Exception:  # JSON cortado: rescatar los textos que haya
        partes = []
        for m in re.findall(r'"text":\s*"((?:[^"\\]|\\.)*)"', raw or ""):
            try:
                partes.append(json.loads('"' + m + '"'))
            except Exception:
                partes.append(m)
        cola = re.search(r'"text":\s*"((?:[^"\\]|\\.)*)$', raw or "")
        if cola:
            partes.append(cola.group(1).replace("\\n", "\n"))
    return "\n\n".join(p for p in partes if p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, choices=["olmocr", "chandra", "dots"])
    ap.add_argument("--api", default="http://localhost:8000/v1")
    ap.add_argument("--hilos", type=int, default=24)
    ap.add_argument("--partes", default="lv,test")
    a = ap.parse_args()
    cli = OpenAI(api_key="EMPTY", base_url=a.api, timeout=1800)
    if a.modelo == "chandra":
        from chandra.model.util import scale_to_fit
        from chandra.prompts import PROMPT_MAPPING
        try:
            from chandra.output import parse_markdown
        except Exception as e:
            print("sin chandra.output:", e, flush=True)
            parse_markdown = None
    trabajos = []
    for parte in a.partes.split(","):
        ent = os.path.join(RAIZ, "datos/gresel", parte)
        sal = os.path.join(RAIZ, "resultados/gresel", a.modelo, parte)
        os.makedirs(sal, exist_ok=True)
        for f in sorted(os.listdir(ent)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                trabajos.append((os.path.join(ent, f), os.path.join(sal, os.path.splitext(f)[0])))
    t0 = time.time()
    hechos = [0]

    def una(par):
        ruta, base = par
        if os.path.exists(base + ".txt"):
            return
        im = Image.open(ruta).convert("RGB")
        t = time.time()
        if a.modelo == "olmocr":
            img, prompt, primero = b64png(lado_largo(im, 1288)), OLMOCR, False
            temps, kw = [0.1, 0.2, 0.3], dict(max_tokens=8000)
        elif a.modelo == "chandra":
            img, prompt, primero = b64png(scale_to_fit(im)), PROMPT_MAPPING["ocr_layout"], True
            temps, kw = [0.0, 0.2, 0.4], dict(max_tokens=12384, top_p=0.1)
        else:
            img, prompt, primero = b64png(dots_fit(im)), DOTS, True
            temps, kw = [0.1, 0.3], dict(max_tokens=16384, top_p=0.9)
        partes = [{"type": "image_url", "image_url": {"url": img}}, {"type": "text", "text": prompt}]
        if not primero:
            partes = partes[::-1]
        raw, fin, intentos = "", None, 0
        for temp in temps:
            intentos += 1
            try:
                r = cli.chat.completions.create(model=a.modelo, messages=[{"role": "user", "content": partes}],
                                                temperature=temp, **kw)
                raw, fin = r.choices[0].message.content or "", r.choices[0].finish_reason
            except Exception as e:
                print("error", os.path.basename(base), type(e).__name__, str(e)[:200], flush=True)
                time.sleep(5)
                continue
            if fin == "stop" and not repite(raw):
                break
        if a.modelo == "olmocr":
            txt = limpiar(sin_front_matter(raw))
        elif a.modelo == "chandra":
            txt = limpiar(parse_markdown(raw) if parse_markdown else raw)
        else:
            txt = limpiar(dots_texto(raw))
        open(base + ".raw", "w").write(raw)
        open(base + ".txt", "w").write(txt)
        json.dump({"fin": fin, "intentos": intentos, "s": round(time.time() - t, 1), "car": len(txt)}, open(base + ".meta.json", "w"))
        hechos[0] += 1
        print(f"{a.modelo} {hechos[0]}/{len(trabajos)} {os.path.basename(base)} fin={fin} int={intentos} {len(txt)} car "
              f"{time.time() - t:.0f}s (total {time.time() - t0:.0f}s)", flush=True)

    with cf.ThreadPoolExecutor(a.hilos) as ex:
        list(ex.map(una, trabajos))
    print("FIN", a.modelo, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
