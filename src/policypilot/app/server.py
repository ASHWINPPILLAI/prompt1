"""
Zero-dependency web server for PolicyPilot.
Replaces Streamlit to run perfectly on Python 3.15 without external binaries.
"""

import http.server
import json
import socketserver
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# --- Native Zero-Dependency .env Loader ---
import os
def load_dotenv():
    env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip().strip("'").strip('"')

load_dotenv()
# ------------------------------------------

from policypilot.pipeline import run_query
from policypilot.prompting.loader import list_prompt_ids

class PolicyPilotHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
            self.wfile.write(html.encode("utf-8"))
        elif self.path == '/api/prompts':
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(list_prompt_ids()).encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/query':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))
            
            question = req.get("question", "")
            prompt_id = req.get("prompt_id", "v0_baseline")
            history = req.get("history", [])
            
            try:
                # Run the backend logic natively
                record = run_query(question, prompt_id, history=history)
                resp = {
                    "answer": record.parsed_output.get("answer") if record.parsed_output else record.raw_output,
                    "tokens": f"{record.prompt_tokens} in / {record.completion_tokens} out",
                    "latency": f"{record.latency_s:.2f}s",
                    "trace": record.react_trace
                }
            except Exception as e:
                resp = {"error": str(e)}

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            
        elif self.path == '/api/query_stream':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))
            
            question = req.get("question", "")
            prompt_id = req.get("prompt_id", "v0_baseline")
            history = req.get("history", [])
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            
            from policypilot.pipeline import stream_query
            try:
                for chunk in stream_query(question, prompt_id, history):
                    self.wfile.write(f"data: {json.dumps({'content': chunk})}\n\n".encode('utf-8'))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as e:
                self.wfile.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode('utf-8'))
                self.wfile.flush()
        else:
            self.send_response(404)
            self.end_headers()

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def run(port=8501):
    with ReusableTCPServer(("", port), PolicyPilotHandler) as httpd:
        print(f"Zero-dependency PolicyPilot Web Server running at http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run(port)
