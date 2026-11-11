import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image

from amanuensis import dots

RESP = [{"bbox": [10, 10, 200, 50], "category": "Title", "text": "# Titular"},
        {"bbox": [10, 60, 200, 140], "category": "Text", "text": "Primera línea\nSegunda línea"},
        {"bbox": [10, 150, 200, 200], "category": "Picture"}]


class _H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        cuerpo = json.dumps({"choices": [{"message": {"content": json.dumps(RESP)}, "finish_reason": "stop"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(cuerpo)

    def log_message(self, *a):
        pass


def test_elementos_cortados():
    raw = '[{"bbox": [1, 2, 3, 4], "category": "Text", "text": "hola"}, {"bbox": [5, 6, 7, 8], "category": "Text", "text": "ad'
    assert dots.elementos(raw)[0]["text"] == "hola"


def test_pagina_con_servidor_falso():
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = dots.ServidorDots(f"http://127.0.0.1:{srv.server_port}/v1")
        s.url = f"http://127.0.0.1:{srv.server_port}"
        p = s.pagina(Image.new("L", (400, 400), 255))
    finally:
        srv.shutdown()
    assert [l.texto for l in p.lineas] == ["Titular", "Primera línea", "Segunda línea"]
    assert abs(p.lineas[0].caja[0] - 10) < 1 and p.modelo == "dots.ocr"


def test_lector_sin_servidor_falla_claro(monkeypatch):
    import pytest

    from amanuensis.motor import Motor

    monkeypatch.delenv("AMANUENSIS_DOTS_URL", raising=False)
    with pytest.raises(SystemExit):
        Motor(lector="dots")


def test_disparador_por_dudas():
    from amanuensis.motor import Motor
    from amanuensis.salidas import Linea, Pagina

    seguro = Pagina(10, 10, [Linea([0, 0, 5, 5], "abcd", confs=[0.99] * 4)])
    dudosa = Pagina(10, 10, [Linea([0, 0, 5, 5], "abcd", confs=[0.2, 0.3, 0.99, 0.99])])
    assert Motor.fraccion_dudas(seguro) == 0 and Motor.fraccion_dudas(dudosa) == 0.5


def test_adaptador_ligero_en_completo():
    import numpy as np

    from amanuensis.completo_ligero import LectorLigeroEnCompleto

    class LL:
        texto, conf, confs, xs = "ab", 0.9, [0.9, 0.8], [90.0, 102.0]

    class Falso:
        def leer(self, pils, sesgo=0.0):
            assert pils[0].size == (32 + 2 * 58, 96 + 2 * 24)  # margen de 0,6 y 0,25 alturas
            return [LL()]

    r = LectorLigeroEnCompleto(Falso()).leer([np.full((96, 32), 255, np.uint8)])
    assert r[0][0] == "ab" and abs(r[0][3][0] - ((90 - 58) / 4 - 0.5)) < 1e-9
