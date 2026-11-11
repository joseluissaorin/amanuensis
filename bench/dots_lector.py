"""dots.ocr sobre las planas de GRESEL, de página entera o como lector de los bloques de Amanuensis (híbrido).

  pagina   página entera con el prompt oficial de maquetación (JSON con cajas, categoría y texto en orden de lectura)
  hibrido  bloques de columna y orden de Amanuensis (modo prensa; salida JSON de `amanu --prensa si -f json,txt`),
           cada bloque recortado con margen y leído por dots.ocr con el prompt de solo texto; se juntan en el orden
           de Amanuensis. Los renglones que no caen en un bloque de columna conservan la lectura de Amanuensis.

Motores: `--motor mlx` (Mac, mlx-vlm, un proceso; lanzar con herramientas/con_gpu.sh) o `--motor api` (servidor con
API de OpenAI, p. ej. vLLM en Vast: `vllm serve rednote-hilab/dots.ocr --served-model-name dots --trust-remote-code`).

  python dots_lector.py pagina  listas/val resultados/dotsmlx_val --n 5
  python dots_lector.py hibrido listas/val resultados/prensa_val resultados/hib_val
  python dots_lector.py hibrido listas/val resultados/prensa_val resultados/hib_val --motor api --api http://localhost:8000/v1 --hilos 24

Salida (la que piden scripts/resumen.py y evaluar.py): <stem>.txt y <stem>.json ({"paginas":[{"lineas":[]}]});
detalle por página en det/<stem>.json (segundos, fichas, y en el híbrido cada bloque con su caja y las dos lecturas,
para probar variantes sin volver a leer: scripts/hibrido_variantes.py).
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
import unicodedata
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Prompts oficiales de dots.ocr (dict_promptmode_to_prompt: prompt_layout_all_en y prompt_ocr); el primero es el mismo
# de eg2-ocr/examen/gresel_cliente.py (constante DOTS), con el que se midieron dots_test y dots_lv.
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
DOTS_TEXTO = "Extract the text content from this image."


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


def repite(t, ventana=300):
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


# ---------------------------------------------------------------- motores
class MotorMLX:
    nombre = "mlx"

    def __init__(self, modelo):
        import mlx.core as mx
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        self.mx, self._gen = mx, generate
        t = time.time()
        self.model, self.proc = load(modelo)
        self.carga = time.time() - t
        self._p = {p: apply_chat_template(self.proc, self.model.config, p, num_images=1) for p in (DOTS, DOTS_TEXTO)}

    def leer(self, im: Image.Image, prompt: str, max_tokens: int, temp: float = 0.1):
        r = self._gen(self.model, self.proc, self._p[prompt], image=[im], max_tokens=max_tokens, temperature=temp, top_p=0.9,
                      verbose=False)
        fin = "length" if r.generation_tokens >= max_tokens else "stop"
        return r.text, fin, {"fichas_ent": r.prompt_tokens, "fichas_sal": r.generation_tokens,
                             "pico_mlx_gb": round(self.mx.get_peak_memory() / 2**30, 2)}

    def limpiar(self):
        self.mx.clear_cache()


class MotorAPI:
    nombre = "api"

    def __init__(self, api, modelo):
        from openai import OpenAI

        self.cli, self.modelo, self.carga = OpenAI(api_key="EMPTY", base_url=api, timeout=1800), modelo, 0.0

    def leer(self, im: Image.Image, prompt: str, max_tokens: int, temp: float = 0.1):
        b = io.BytesIO()
        im.save(b, "PNG")
        url = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
        r = self.cli.chat.completions.create(
            model=self.modelo, temperature=temp, top_p=0.9, max_tokens=max_tokens,
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}])
        u = r.usage
        return r.choices[0].message.content or "", r.choices[0].finish_reason, {
            "fichas_ent": getattr(u, "prompt_tokens", 0), "fichas_sal": getattr(u, "completion_tokens", 0)}

    def limpiar(self):
        pass


def leer_con_reintento(motor, im, prompt, max_tokens):
    raw, fin, info, intentos = "", None, {}, 0
    for temp in (0.1, 0.3):
        intentos += 1
        try:
            raw, fin, info = motor.leer(im, prompt, max_tokens, temp)
        except Exception as e:  # noqa: BLE001
            print("  error", type(e).__name__, str(e)[:200], flush=True)
            if motor.nombre == "mlx":
                raise
            time.sleep(5)
            continue
        if fin == "stop" and not repite(raw):
            break
    info["intentos"] = intentos
    return raw, fin, info


# ---------------------------------------------------------------- híbrido
def grupos_de_amanuensis(pag: dict) -> list[dict]:
    """Trozos de la página en el orden de lectura de Amanuensis: {"lineas": [...], "caja": [...] | None, "texto": str}.
    Un trozo con caja es un bloque de columna del modo prensa (zona con "prensa": true) y se le da a dots.ocr; los demás
    (tablas, renglones sueltos sin bloque) conservan la lectura de Amanuensis. Se siguen los «bloques» del JSON, que son
    lo que sale en el .txt; los de rol «figura» no salen (tampoco en el .txt de Amanuensis)."""
    ls, zs = pag["lineas"], pag.get("zonas", [])
    orden = [i for b in pag.get("bloques", []) if b.get("rol") != "figura" for i in b["lineas"]] or list(range(len(ls)))
    corte = set()  # frontera de bloque de salida (párrafo): se respeta como salto dentro del texto de Amanuensis
    for b in pag.get("bloques", []):
        if b.get("rol") != "figura" and b["lineas"]:
            corte.add(b["lineas"][0])
    out = []
    for i in orden:
        z = ls[i].get("zona", -1)
        pr = 0 <= z < len(zs) and bool(zs[z].get("prensa"))
        clave = z if pr else None
        if out and out[-1]["zona"] == clave and clave is not None:
            out[-1]["lineas"].append(i)
        else:
            out.append({"zona": clave, "lineas": [i]})
    for g in out:
        cs = [ls[i]["caja"] for i in g["lineas"]]
        g["caja"] = [min(c[0] for c in cs), min(c[1] for c in cs), max(c[2] for c in cs), max(c[3] for c in cs)] if g["zona"] is not None else None
        trozos, act = [], []
        for i in g["lineas"]:
            if act and i in corte:
                trozos.append("\n".join(act))
                act = []
            act.append(ls[i]["texto"])
        trozos.append("\n".join(act))
        g["texto"] = "\n\n".join(trozos)
        g["alto_renglon"] = float(sorted(c[3] - c[1] for c in cs)[len(cs) // 2])
    return out


def recorte(im: Image.Image, caja, alto_renglon: float, margen: float = 0.35, ampliar: float = 1.0, min_alto: int = 0):
    m = margen * alto_renglon
    x0, y0 = max(0, int(caja[0] - m)), max(0, int(caja[1] - m))
    x1, y1 = min(im.size[0], int(caja[2] + m + 1)), min(im.size[1], int(caja[3] + m + 1))
    r = im.crop((x0, y0, x1, y1))
    z = ampliar
    if min_alto and alto_renglon * z < min_alto:  # letra pequeña: que el renglón tenga al menos min_alto píxeles
        z = min(4.0, min_alto / max(1.0, alto_renglon))
    if z != 1.0:
        r = r.resize((max(28, round(r.size[0] * z)), max(28, round(r.size[1] * z))), Image.LANCZOS)
    return r


def json_amanuensis(carpeta: Path, stem: str):
    for f in (stem, unicodedata.normalize("NFC", stem), unicodedata.normalize("NFD", stem)):
        p = carpeta / f"{f}.json"
        if p.exists():
            return json.load(open(p))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["pagina", "hibrido"])
    ap.add_argument("lista", help="carpeta de imágenes (listas/val…)")
    ap.add_argument("resto", nargs="+", help="pagina: SALIDA; hibrido: CARPETA_AMANUENSIS SALIDA")
    ap.add_argument("--motor", default="mlx", choices=["mlx", "api"])
    ap.add_argument("--modelo", default=None, help="mlx: repo o carpeta (mlx-community/dots.ocr-8bit); api: nombre servido (dots)")
    ap.add_argument("--api", default="http://localhost:8000/v1")
    ap.add_argument("--hilos", type=int, default=1, help="solo con --motor api")
    ap.add_argument("--n", type=int, default=0, help="solo las n primeras páginas")
    ap.add_argument("--solo", default=None, help="fichero con un stem por renglón (subconjunto)")
    ap.add_argument("--max-pixels", type=int, default=11289600, help="tope de píxeles de la imagen que ve dots.ocr")
    ap.add_argument("--margen", type=float, default=0.35, help="híbrido: margen del recorte, en altos de renglón")
    ap.add_argument("--ampliar", type=float, default=1.0, help="híbrido: factor de ampliación del recorte")
    ap.add_argument("--min-alto", type=int, default=0, help="híbrido: amplía el recorte hasta que el renglón mida esto (px)")
    a = ap.parse_args()
    lista = Path(a.lista)
    amanu = Path(a.resto[0]) if a.modo == "hibrido" else None
    sal = Path(a.resto[-1])
    (sal / "det").mkdir(parents=True, exist_ok=True)
    fs = sorted(f for f in os.listdir(lista) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif")))
    if a.solo:
        quiero = {unicodedata.normalize("NFC", s.strip()) for s in open(a.solo) if s.strip()}
        fs = [f for f in fs if unicodedata.normalize("NFC", Path(f).stem) in quiero]
    if a.n:
        fs = fs[: a.n]
    motor = MotorMLX(a.modelo or "mlx-community/dots.ocr-8bit") if a.motor == "mlx" else MotorAPI(a.api, a.modelo or "dots")
    print(f"motor {motor.nombre} listo en {motor.carga:.1f} s; {len(fs)} páginas", flush=True)
    t0 = time.time()

    def una(f):
        stem = Path(f).stem
        if (sal / f"{stem}.txt").exists():
            return
        im = Image.open(lista / f).convert("RGB")
        t = time.time()
        det = {"pag": stem, "ancho": im.size[0], "alto": im.size[1], "motor": motor.nombre}
        if a.modo == "pagina":
            img = dots_fit(im, max_pixels=a.max_pixels)
            raw, fin, info = leer_con_reintento(motor, img, DOTS, 16384)
            txt = limpiar(dots_texto(raw))
            det.update(info, fin=fin, entrada=list(img.size), raw=raw)
        else:
            d = json_amanuensis(amanu, stem)
            if d is None:
                print("  sin JSON de Amanuensis:", stem, flush=True)
                return
            gs = grupos_de_amanuensis(d["paginas"][0])
            partes, n_dots, f_ent, f_sal = [], 0, 0, 0

            def bloque(g):
                if g["caja"] is None or not g["texto"].strip():
                    return dict(g, dots=None)
                r = dots_fit(recorte(im, g["caja"], g["alto_renglon"], a.margen, a.ampliar, a.min_alto), max_pixels=a.max_pixels)
                tb = time.time()
                raw, fin, info = leer_con_reintento(motor, r, DOTS_TEXTO, int(200 + 1.2 * len(g["texto"])))
                return dict(g, dots=limpiar(raw), fin=fin, s=round(time.time() - tb, 2), entrada=list(r.size), **info)

            if motor.nombre == "api" and a.hilos > 1:
                with cf.ThreadPoolExecutor(min(8, a.hilos)) as ex:
                    hechos = list(ex.map(bloque, gs))
            else:
                hechos = [bloque(g) for g in gs]
            for g in hechos:
                if g.get("dots") is not None:
                    n_dots += 1
                    f_ent += g.get("fichas_ent", 0)
                    f_sal += g.get("fichas_sal", 0)
                    # salida vacía, cortada o en bucle: se queda la lectura de Amanuensis
                    mala = not g["dots"].strip() or g["fin"] != "stop" or repite(g["dots"])
                    g["usa"] = "amanuensis" if mala else "dots"
                else:
                    g["usa"] = "amanuensis"
                partes.append(g["texto"] if g["usa"] == "amanuensis" else g["dots"])
            txt = "\n\n".join(p for p in partes if p.strip())
            det.update(bloques=hechos, n_bloques=len(hechos), n_dots=n_dots, fichas_ent=f_ent, fichas_sal=f_sal,
                       pico_mlx_gb=max([g.get("pico_mlx_gb", 0) for g in hechos] or [0]))
        det["s"] = round(time.time() - t, 2)
        motor.limpiar()
        json.dump(det, open(sal / "det" / f"{stem}.json", "w"), ensure_ascii=False)
        json.dump({"paginas": [{"lineas": []}]}, open(sal / f"{stem}.json", "w"))
        open(sal / f"{stem}.txt", "w").write(txt)
        print(f"{stem}: {det['s']:.1f} s, {len(txt)} car, fichas {det.get('fichas_ent', 0)}+{det.get('fichas_sal', 0)}"
              + (f", {det['n_dots']}/{det['n_bloques']} bloques con dots" if a.modo == "hibrido" else f", fin={det.get('fin')}")
              + (f", pico MLX {det.get('pico_mlx_gb')} GB" if motor.nombre == "mlx" else "") + f" (total {time.time() - t0:.0f} s)", flush=True)

    if motor.nombre == "api" and a.hilos > 1:
        with cf.ThreadPoolExecutor(max(1, a.hilos // 8) if a.modo == "hibrido" else a.hilos) as ex:
            list(ex.map(una, fs))
    else:
        for f in fs:
            una(f)
    print(f"FIN {a.modo} {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
