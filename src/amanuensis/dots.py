"""dots.ocr como lector opcional de página entera, por un servidor con API de OpenAI (p. ej. vLLM:
`vllm serve rednote-hilab/dots.ocr --served-model-name dots --trust-remote-code`). No es el camino por defecto: se
activa con `--lector dots --servidor URL` (o AMANUENSIS_DOTS_URL) y no necesita ninguna dependencia nueva (urllib).

En el corpus de prensa de GRESEL lee mejor que Amanuensis (CER de página 5,08 frente a 9,60 en validación), pero pesa
unos 6 GB en GPU de servidor y no cabe en el Mac. Se manda la página entera con el prompt oficial de maquetación; la
respuesta (JSON con caja, categoría y texto en Markdown, en orden de lectura) se convierte en renglones de Amanuensis:
cada línea de texto de un elemento recibe un trozo proporcional de su caja.
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import time
import urllib.request

from PIL import Image

from .salidas import Linea, Pagina

PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

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
MAX_PIXELS = 11289600


def _ajustar(im: Image.Image, factor: int = 28):
    w, h = im.size
    s = (MAX_PIXELS / (w * h)) ** 0.5 if w * h > MAX_PIXELS else 1.0
    nw, nh = max(factor, int(w * s // factor * factor)), max(factor, int(h * s // factor * factor))
    return (im if (nw, nh) == (w, h) else im.resize((nw, nh), Image.LANCZOS)), nw, nh


def limpiar(t: str) -> str:
    t = re.sub(r"<br\s*/?>", "\n", t or "")
    t = re.sub(r"</(p|div|tr|li|h\d)>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"^```\w*|```$", "", t, flags=re.M)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"\*\*|__", "", t)
    return re.sub(r"\n{2,}", "\n", re.sub(r"[ \t]+\n", "\n", t)).strip()


def elementos(raw: str) -> list[dict]:
    """JSON de dots.ocr -> [{"bbox", "category", "text"}]; si la respuesta está cortada, rescata los textos que haya."""
    try:
        el = json.loads(raw)
        if isinstance(el, dict):
            el = el.get("layout") or el.get("elements") or list(el.values())[0]
        return [e for e in el if isinstance(e, dict)]
    except Exception:  # noqa: BLE001
        out = []
        for m in re.finditer(r'\{[^{}]*?"bbox":\s*\[([^\]]*)\][^{}]*?"category":\s*"([^"]*)"(?:[^{}]*?"text":\s*"((?:[^"\\]|\\.)*)")?', raw or ""):
            try:
                bb = [float(v) for v in m.group(1).split(",")]
                tx = json.loads('"' + (m.group(3) or "") + '"')
            except Exception:  # noqa: BLE001
                continue
            out.append({"bbox": bb, "category": m.group(2), "text": tx})
        return out


class ServidorDots:
    def __init__(self, url: str, modelo: str = "dots", max_tokens: int = 16384, timeout: int = 1800):
        self.url = url.rstrip("/")
        self.modelo, self.max_tokens, self.timeout = modelo, max_tokens, timeout

    def _pedir(self, im: Image.Image, temp: float) -> tuple[str, str]:
        b = io.BytesIO()
        im.save(b, "PNG")
        cuerpo = {"model": self.modelo, "temperature": temp, "top_p": 0.9, "max_tokens": self.max_tokens,
                  "messages": [{"role": "user", "content": [
                      {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}},
                      {"type": "text", "text": PROMPT}]}]}
        req = urllib.request.Request(self.url + "/chat/completions", json.dumps(cuerpo).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            j = json.load(r)
        c = j["choices"][0]
        return c["message"].get("content") or "", c.get("finish_reason", "")

    def comprobar(self):
        try:
            urllib.request.urlopen(self.url + "/models", timeout=15).read()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"no llego al servidor de dots.ocr ({self.url}): {e}")

    def pagina(self, img: Image.Image) -> Pagina:
        t0 = time.time()
        W0, H0 = img.size
        im, nw, nh = _ajustar(img.convert("RGB"))
        raw = ""
        for temp in (0.1, 0.3):  # un reintento más templado si se corta o se repite
            raw, fin = self._pedir(im, temp)
            if fin == "stop":
                break
        sx, sy = W0 / nw, H0 / nh
        lineas = []
        for e in elementos(raw):
            if e.get("category") == "Picture" or not e.get("text"):
                continue
            bb = e.get("bbox") or [0, 0, nw, nh]
            x0, y0, x1, y1 = bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy
            ts = [t for t in limpiar(e["text"]).split("\n") if t.strip()]
            for k, t in enumerate(ts):
                a, b = y0 + (y1 - y0) * k / len(ts), y0 + (y1 - y0) * (k + 1) / len(ts)
                lineas.append(Linea([round(x0, 1), round(a, 1), round(x1, 1), round(b, 1)], t.strip(), 1.0,
                                    zona_clase=e.get("category")))
        p = Pagina(W0, H0, lineas, "ocr (dots.ocr, servidor)", "dots.ocr")
        p.segundos = time.time() - t0
        return p


def desde_entorno(url: str | None = None) -> ServidorDots | None:
    url = url or os.environ.get("AMANUENSIS_DOTS_URL")
    return ServidorDots(url, os.environ.get("AMANUENSIS_DOTS_MODELO", "dots")) if url else None
