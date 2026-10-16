"""Stand-in for llama-server / llama-perplexity so the eval plumbing is testable without a GPU."""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


def server(port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send({"status": "ok"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            q = body["messages"][-1]["content"]
            self._send({"choices": [{"message": {"content": "4" if "2+2" in q else "no idea"}}]})

    HTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    if sys.argv[1] == "ppl":
        print("Final estimate: PPL = 7.1234 +/- 0.05000", flush=True)
    else:
        server(int(sys.argv[sys.argv.index("--port") + 1]))
