#!/usr/bin/env python3
"""Dev server for the ALE docs site.

Plain `python -m http.server` lets the browser cache nav.js / app.js, so edits
don't show up without a hard refresh. This server sends no-cache headers on
every response, so a normal reload always picks up the latest files.

    python3 serve.py [port]      # default 5500
"""
import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5500


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


class Server(socketserver.TCPServer):
    allow_reuse_address = True


with Server(("", PORT), NoCacheHandler) as httpd:
    print(f"ale-docs-site → http://localhost:{PORT}/  (no-cache; Ctrl-C to stop)")
    httpd.serve_forever()
