import http.server, time, json
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_GET(self):
        if self.path == "/ping":
            b = b"pong"; self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        self.send_response(200); self.send_header("Content-Type","text/event-stream"); self.send_header("Transfer-Encoding","chunked"); self.end_headers()
        for i in range(5):
            d = f"data: {json.dumps({'i': i})}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(d), d)); self.wfile.flush(); time.sleep(1)
        self.wfile.write(b"0\r\n\r\n")
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
