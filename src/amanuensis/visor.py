"""Visor local: amanuensis visor [--puerto 8766]. Arrastra un PDF o una imagen; muestra la página con las
líneas y su orden, y el texto. Nada sale del ordenador."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HTML = """<!doctype html><html lang="es"><meta charset="utf-8"><title>amanuensis</title>
<style>body{font:15px Georgia,serif;margin:0;background:#f6f1e7;color:#2b2622}header{padding:12px 20px;border-bottom:1px solid #d8cfbf}
#zona{border:2px dashed #b5562f;padding:24px;margin:16px 20px;text-align:center;cursor:pointer}
main{display:flex;gap:16px;padding:0 20px 20px}#img{position:relative;flex:1}#img img{width:100%}
#img div{position:absolute;border:1px solid #b5562f88;background:#b5562f14;font-size:10px;color:#b5562f}
pre{flex:1;white-space:pre-wrap;background:#fffdf8;padding:12px;border:1px solid #d8cfbf;max-height:85vh;overflow:auto}</style>
<header><b>amanuensis</b> · visor local</header><div id="zona">Arrastra aquí un PDF o una imagen (o pulsa para elegir)</div>
<input type="file" id="f" hidden><main><div id="img"></div><pre id="txt"></pre></main>
<script>
const z=document.getElementById('zona'),f=document.getElementById('f');
z.onclick=()=>f.click();f.onchange=()=>enviar(f.files[0]);
z.ondragover=e=>e.preventDefault();z.ondrop=e=>{e.preventDefault();enviar(e.dataTransfer.files[0])};
async function enviar(a){z.textContent='Leyendo '+a.name+'…';
 const r=await fetch('/leer?nombre='+encodeURIComponent(a.name),{method:'POST',body:a});const d=await r.json();
 z.textContent=a.name+' · '+d.paginas.length+' páginas';const p=d.paginas[0],c=document.getElementById('img');
 c.innerHTML='<img src="data:image/png;base64,'+d.imagen+'">';
 p.lineas.forEach((l,i)=>{const b=document.createElement('div');const [x0,y0,x1,y1]=l.caja;
  Object.assign(b.style,{left:100*x0/p.ancho+'%',top:100*y0/p.alto+'%',width:100*(x1-x0)/p.ancho+'%',height:100*(y1-y0)/p.alto+'%'});
  b.textContent=i+1;b.title=l.texto+' ('+l.conf+')';c.appendChild(b)});
 document.getElementById('txt').textContent=d.md}
</script></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="amanuensis visor")
    ap.add_argument("--puerto", type=int, default=8766)
    a = ap.parse_args(argv)
    from PIL import Image

    from .motor import Motor

    motor = Motor(verbose=True)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(HTML.encode())

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            datos = self.rfile.read(n)
            nombre = os.path.basename(self.path.split("nombre=")[-1]) or "entrada"
            with tempfile.TemporaryDirectory() as d:
                r = os.path.join(d, nombre)
                open(r, "wb").write(datos)
                doc = motor.leer(r, 1)
                if datos[:5] == b"%PDF-":
                    from . import pdf

                    im = pdf.imagen(pdf.abrir(r)[0], motor.dpi)
                else:
                    im = Image.open(r).convert("L")
            b = io.BytesIO(); im.save(b, "PNG")
            js = json.loads(doc.exportar("json"))
            js["md"] = doc.exportar("md")
            js["imagen"] = base64.b64encode(b.getvalue()).decode()
            s = json.dumps(js, ensure_ascii=False).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(s)

    print(f"visor en http://127.0.0.1:{a.puerto}/ (Ctrl+C para salir)")
    ThreadingHTTPServer(("127.0.0.1", a.puerto), H).serve_forever()
