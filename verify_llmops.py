import requests
import json
import time

url = "http://localhost:8085/hackrx/run"
payload = {
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", # This one was empty, let's try another
    "documents": "https://raw.githubusercontent.com/sugarshack/JBBR-Backend/main/stored_pdfs/policy.pdf", # Guessing this exists relative to some known repo or just use a known good one
    "documents": "https://www.irs.gov/pub/irs-pdf/fw4.pdf", # IRS W-4 is a standard rich PDF
    "questions": ["What is the purpose of this form?", "What is the exemption limit?"],
    "session_id": "verify-session-1",
    "user_id": "verify-user-1"
}

print(f"🚀 Sending request to {url}...")
try:
    response = requests.post(url, json=payload, timeout=60)
    print(f"✅ Status Code: {response.status_code}")
    print("📝 Response Body:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"❌ Request failed: {e}")

print("\n📊 Checking /metrics for updates...")
time.sleep(2)
metrics = requests.get("http://localhost:8085/metrics").text
for line in metrics.split("\n"):
    if "rag_" in line and not line.startswith("#"):
        print(f"  {line}")
