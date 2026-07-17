from fastapi.testclient import TestClient
from src.api import app
import os
import time

# Disable HyDE for faster testing
os.environ["USE_HYDE"] = "false"

print("Initializing TestClient (This will trigger the 2-second startup warmup)...")
with TestClient(app) as client:
    print("\n--- Sending Question 1 ---")
    payload1 = {"question": "What is the Public Investment Fund (PIF)?"}
    
    t0 = time.time()
    response1 = client.post("/ask", json=payload1)
    data1 = response1.json()
    
    print(f"Answer: {data1.get('answer', 'ERROR')}")
    session_id = data1.get("session_id")
    print(f"Session ID acquired: {session_id}")

    print("\n--- Sending Follow-up Question 2 ---")
    # Using 'its' forces the LLM to look at the memory to know we mean the PIF
    payload2 = {"question": "What is its targeted asset value for the year 2030?", "session_id": session_id}
    
    response2 = client.post("/ask", json=payload2)
    data2 = response2.json()
    
    print(f"Answer: {data2.get('answer', 'ERROR')}")
    print("\nMemory integration test complete.")