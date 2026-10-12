"""Tiny static server that injects BACKEND_URL into index.html."""

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
PORT = int(os.getenv("PORT", "8080"))


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open("index.html", "r", encoding="utf-8") as f:
                    html = f.read()
                html = html.replace("__BACKEND_URL__", BACKEND_URL)
                data = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except FileNotFoundError:
                self.send_error(404)
                return
        return super().do_GET()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving on :{PORT} with BACKEND_URL={BACKEND_URL}")
    server.serve_forever()
