import urllib.request
import json

req = urllib.request.Request(
    'http://127.0.0.1:8502/api/chat',
    data=json.dumps({"question": "hello", "version": "v0_baseline"}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as res:
        print(res.status)
        print(res.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    print(e.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
