import urllib.request
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

QUESTIONS = [
    # --- 20 Mathematical ---
    "What is 256 * 128?",
    "If a person receives $1200 a month, what is their annual income?",
    "Calculate the compound interest on $10,000 at 5% annually for 10 years.",
    "What is the square root of 144?",
    "Solve for x: 3x + 15 = 45",
    "If I travel 60 miles per hour for 4.5 hours, how far do I travel?",
    "What is 15% of 850?",
    "If a pizza has 8 slices and I eat 3, what fraction is left?",
    "What is the derivative of x^2 + 3x?",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "If a 20% discount saves me $15, what was the original price?",
    "What is 2^10?",
    "If a rectangular room is 12ft by 15ft, what is its area?",
    "What is the probability of rolling a 7 with two six-sided dice?",
    "Solve for y: 2y - 10 = 0",
    "How many seconds are in a 365-day year?",
    "If I buy 3 apples for $1.20 each, how much change from a $5 bill?",
    "What is the sum of the first 100 positive integers?",
    "If the radius of a circle is 5cm, what is its circumference?",
    "What is 45 divided by 1.5?",

    # --- 20 Analytical ---
    "Compare and contrast the economic policies of Keynes and Hayek.",
    "What are the primary causes of inflation?",
    "Analyze the impact of artificial intelligence on the job market.",
    "Why do governments implement tariffs?",
    "Explain the concept of supply and demand with an example.",
    "What are the ethical implications of autonomous vehicles?",
    "How does a central bank control interest rates?",
    "Evaluate the pros and cons of universal basic income.",
    "Why did the housing market crash in 2008?",
    "What is the prisoner's dilemma in game theory?",
    "How does climate change impact global supply chains?",
    "Analyze the effectiveness of remote work vs office work.",
    "What are the main arguments for and against nuclear energy?",
    "How does social media algorithm design affect political polarization?",
    "What is the difference between a recession and a depression?",
    "Evaluate the impact of demographic aging on healthcare systems.",
    "Why do monopolies harm consumer welfare?",
    "How does cryptocurrency challenge traditional banking?",
    "What is the relationship between education level and lifetime earnings?",
    "Analyze the factors that lead to urban gentrification.",

    # --- 10 Normal Facts / News ---
    "What is the capital of Australia?",
    "Who wrote the play Hamlet?",
    "What is the speed of light in a vacuum?",
    "Which planet is closest to the sun?",
    "When did the Berlin Wall fall?",
    "Who is the current President of France?",
    "What is the deepest ocean trench on Earth?",
    "What chemical element has the symbol Au?",
    "In what year was the first iPhone released?",
    "What is the tallest mountain in the world?"
]

def ask_question(q, idx):
    req_data = json.dumps({
        "question": q,
        "prompt_id": "v5_react",  # Using the advanced ReAct version
        "history": []
    }).encode('utf-8')

    req = urllib.request.Request(
        'http://127.0.0.1:8501/api/query_stream',
        data=req_data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            full_ans = ""
            for line in response:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if "content" in data:
                            full_ans += data["content"]
                    except:
                        pass
            
            latency = time.time() - start
            # Clean up <think> tags for logging
            import re
            clean_ans = re.sub(r'<think>[\s\S]*?(<\/think>|$)', '', full_ans).strip()
            
            # Print excerpt
            excerpt = clean_ans[:100].replace('\n', ' ') + ('...' if len(clean_ans) > 100 else '')
            return f"[{idx:02d}] PASS ({latency:.1f}s) Q: {q[:30]:<30} | A: {excerpt}"
    except Exception as e:
        return f"[{idx:02d}] FAIL ({(time.time() - start):.1f}s) Q: {q[:30]:<30} | ERROR: {e}"

if __name__ == '__main__':
    print(f"Starting Bulk Test of {len(QUESTIONS)} questions (Sequential to avoid Free API Rate Limits)...", flush=True)
    results = []
    
    for i, q in enumerate(QUESTIONS):
        res = ask_question(q, i+1)
        print(res, flush=True)
        results.append(res)
        time.sleep(1.5)  # Avoid rate limits
            
    fails = sum(1 for r in results if 'FAIL' in r)
    print(f"\n--- BULK TEST SUMMARY ---", flush=True)
    print(f"Total: {len(QUESTIONS)} | Passed: {len(QUESTIONS) - fails} | Failed: {fails}", flush=True)
