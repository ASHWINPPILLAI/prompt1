import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("src").resolve()))

def run_tests():
    print("Running Regression Suite...")
    passes = 0
    failures = 0
    
    try:
        from policypilot.prompting.loader import list_prompt_ids
        prompts = list_prompt_ids()
        assert len(prompts) > 0, "No valid prompts found!"
        assert "autocot_demos" not in prompts, "Empty prompt autocot_demos was not filtered out!"
        passes += 1
        print("[PASS] Prompt loader correctly filters empty files.")
    except Exception as e:
        failures += 1
        print(f"[FAIL] Prompt loader: {e}")
        
    try:
        from policypilot.retrieval import retrieve
        res = retrieve("minimum age limit")
        assert len(res) > 0, "Retrieve returned empty list."
        assert "text" in res[0], "Retrieve missing text field."
        passes += 1
        print("[PASS] SQLite FTS5 Search Engine works.")
    except Exception as e:
        failures += 1
        print(f"[FAIL] Retrieval: {e}")
        
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8501/")
        with urllib.request.urlopen(req) as response:
            assert response.status == 200
        passes += 1
        print("[PASS] Server is running and serves UI.")
    except Exception as e:
        failures += 1
        print(f"[FAIL] Server HTTP: {e}")
        
    print(f"\nResults: {passes} passed, {failures} failed.")
    sys.exit(failures)

if __name__ == "__main__":
    run_tests()
