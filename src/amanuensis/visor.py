"""Visor local: amanuensis visor [--puerto 8766]. Arrastra un PDF o una imagen; muestra la página con las
líneas y su orden, y el texto con las letras dudosas resaltadas (pasa el ratón para ver su probabilidad).
Deslizadores de «arrojo» (0-1, 0,5 = neutro) y de umbral de duda: al moverlos se vuelve a leer el mismo fichero.
Nada sale del ordenador."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HTML = """<!doctype html><html lang="es"><meta charset="utf-8"><title>Amanuensis · visor local</title>
<style>body{font:15px Georgia,serif;margin:0;background:#f6f1e7;color:#2b2622}
header{padding:12px 20px;border-bottom:1px solid #d8cfbf;display:flex;gap:28px;align-items:center;flex-wrap:wrap}
label{font-size:13px}input[type=range]{vertical-align:middle;accent-color:#b5562f}output{display:inline-block;width:2.6em}
#zona{border:2px dashed #b5562f;padding:24px;margin:16px 20px;text-align:center;cursor:pointer}
main{display:flex;gap:16px;padding:0 20px 20px}#img{position:relative;flex:1}#img img{width:100%}
#img div{position:absolute;border:1px solid #b5562f88;background:#b5562f14;font-size:10px;color:#b5562f}
#img div.floja{border-color:#8a1c1c;background:#8a1c1c22}
#txt{flex:1;white-space:pre-wrap;background:#fffdf8;padding:12px;border:1px solid #d8cfbf;max-height:85vh;overflow:auto;line-height:1.5}
mark{background:#f1c7b3;color:inherit;border-bottom:2px solid #b5562f;cursor:help}mark.muy{background:#e79a7c}
small{color:#7a6f63}</style>
<header><b>Amanuensis</b><span>visor local</span>
<label title="Cuánto se arriesga el lector: más alto en escaneos muy malos o de baja resolución (calla menos letras); más bajo si prefieres huecos a inventos">
Arrojo <input type="range" id="arrojo" min="0" max="1" step="0.05" value="0.5"> <output id="va">0,50</output></label>
<label title="Las letras con probabilidad menor que este valor se resaltan como dudosas">
Duda bajo <input type="range" id="duda" min="0" max="0.95" step="0.05" value="0.5"> <output id="vd">0,50</output></label>
<small id="estado"></small></header>
<div id="zona">Arrastra aquí un PDF o una imagen (o pulsa para elegir)</div>
<input type="file" id="f" hidden><main><div id="img"></div><div id="txt"></div></main>
<script>
const $=id=>document.getElementById(id),z=$('zona'),f=$('f');let fichero=null,datos=null;
const coma=v=>Number(v).toFixed(2).replace('.',',');
z.onclick=()=>f.click();f.onchange=()=>{fichero=f.files[0];enviar()};
z.ondragover=e=>e.preventDefault();z.ondrop=e=>{e.preventDefault();fichero=e.dataTransfer.files[0];enviar()};
$('arrojo').oninput=()=>$('va').textContent=coma($('arrojo').value);$('arrojo').onchange=()=>fichero&&enviar();
$('duda').oninput=()=>{$('vd').textContent=coma($('duda').value);datos&&pintar()};
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function enviar(){if(!fichero)return;z.textContent='Leyendo '+fichero.name+'…';
 const t0=performance.now();
 const r=await fetch('/leer?nombre='+encodeURIComponent(fichero.name)+'&arrojo='+$('arrojo').value,{method:'POST',body:fichero});
 datos=await r.json();z.textContent=fichero.name+' · '+datos.paginas.length+' páginas';
 $('estado').textContent='arrojo '+coma(datos.arrojo)+' · '+((performance.now()-t0)/1000).toFixed(1).replace('.',',')+' s';pintar()}
function pintar(){const u=Number($('duda').value),p=datos.paginas[0],c=$('img');let n=0;
 c.innerHTML='<img src="data:image/png;base64,'+datos.imagen+'">';
 const html=p.lineas.map((l,i)=>{const [x0,y0,x1,y1]=l.caja,b=document.createElement('div');
  Object.assign(b.style,{left:100*x0/p.ancho+'%',top:100*y0/p.alto+'%',width:100*(x1-x0)/p.ancho+'%',height:100*(y1-y0)/p.alto+'%'});
  b.textContent=i+1;b.title=l.texto+' (confianza '+coma(l.conf)+')';if(l.conf<u)b.className='floja';c.appendChild(b);
  const cs=l.confs&&l.confs.length===l.texto.length?l.confs:null;if(!cs)return esc(l.texto);
  let s='',k=0;const t=[...l.texto];
  while(k<t.length){const d=cs[k]<u&&t[k].trim();let j=k,m=1;
   while(j<t.length&&((cs[j]<u&&t[j].trim())?true:false)===!!d){if(d)m=Math.min(m,cs[j]);j++}
   const tr=esc(t.slice(k,j).join(''));
   if(d){n+=j-k;s+='<mark class="'+(m<u/2?'muy':'')+'" title="probabilidad '+coma(m)+'">'+tr+'</mark>'}else s+=tr;k=j}
  return s}).join('\\n');
 $('txt').innerHTML=html||'<small>(sin texto)</small>';
 $('estado').textContent=$('estado').textContent.split(' · dudosas')[0]+' · dudosas: '+n+' letras'}
</script></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="amanuensis visor")
    ap.add_argument("--puerto", type=int, default=8766)
    a = ap.parse_args(argv)
    from PIL import Image

    from .lector import sesgo_blanco
    from .motor import Motor

    motor = Motor(verbose=True)
    cerrojo = threading.Lock()

    def poner_arrojo(v: float):
        v = min(1.0, max(0.0, v))
        motor.arrojo, motor.sesgo = v, sesgo_blanco(v)
        if motor._completo is not None:
            motor._completo.arrojo = v
        return v

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(HTML.encode())

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            datos = self.rfile.read(n)
            q = parse_qs(urlparse(self.path).query)
            nombre = os.path.basename(q.get("nombre", ["entrada"])[0]) or "entrada"
            with cerrojo, tempfile.TemporaryDirectory() as d:
                arrojo = poner_arrojo(float(q.get("arrojo", ["0.5"])[0]))
                r = os.path.join(d, nombre)
                open(r, "wb").write(datos)
                doc = motor.leer(r, 1)
                if datos[:5] == b"%PDF-":
                    from . import pdf

                    im = pdf.imagen(pdf.abrir(r)[0], motor.dpi)
                else:
                    from .geometria_pagina import abrir

                    im = abrir(r).convert("L")  # con la orientación EXIF aplicada
            b = io.BytesIO(); im.save(b, "PNG")
            js = json.loads(doc.exportar("json"))
            js["md"] = doc.exportar("md")
            js["arrojo"] = arrojo
            js["imagen"] = base64.b64encode(b.getvalue()).decode()
            s = json.dumps(js, ensure_ascii=False).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(s)

        def log_message(self, *a):
            pass

    print(f"visor en http://127.0.0.1:{a.puerto}/ (Ctrl+C para salir)")
    ThreadingHTTPServer(("127.0.0.1", a.puerto), H).serve_forever()
