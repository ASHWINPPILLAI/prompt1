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
            
        elif self.path == '/api/chat':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                req = json.loads(post_data.decode('utf-8'))
                
                question = req.get("question", "")
                version = req.get("version", "v0_baseline")
                params = req.get("params", {})
                history = req.get("history", [])
                
                from policypilot.pipeline import run_query
                record = run_query(question, version, config=params, history=history)
                
                # Format the RunRecord to match the new frontend's API contract
                answer = record.raw_output
                if record.parsed_output:
                    answer = record.parsed_output.get("answer", record.raw_output)
                
                status = "unverified"
                if record.retrieved_section_ids:
                    status = "grounded"
                    
                resp_data = {
                    "answer": answer,
                    "answerable": True,
                    "status": status,
                    "version": record.prompt_id,
                    "citations": [{"section_id": cid} for cid in record.retrieved_section_ids],
                    "reasoning": record.parsed_output.get("reasoning", []) if record.parsed_output else [],
                    "react_trace": [{"thought": step.get("thought", ""), "action": f"{step.get('action', '')}[{step.get('action_input', '')}]", "observation": step.get("observation", "")} for step in record.parsed_output.get("trace", [])] if record.parsed_output else [],
                    "verifier": None,
                    "router": None,
                    "usage": {
                        "prompt_tokens": getattr(record, 'prompt_tokens', 0) or 0,
                        "completion_tokens": getattr(record, 'completion_tokens', 0) or 0,
                        "latency_s": round(getattr(record, 'latency_s', 0.1) or 0.1, 2)
                    }
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(resp_data).encode('utf-8'))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

def run(port=8501):
    with ReusableThreadingTCPServer(("", port), PolicyPilotHandler) as httpd:
        print(f"Zero-dependency PolicyPilot Web Server running at http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run(port)
