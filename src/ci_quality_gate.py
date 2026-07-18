import os
import json
import sys
from datetime import datetime

def main():
    scores_file = os.path.join("data", "evaluation", "evaluation_scores.json")
    history_file = os.path.join("data", "evaluation", "scores_history.json")
    
    # 1. Read latest execution scores
    if not os.path.exists(scores_file):
        print(f"Error: {scores_file} not found.")
        sys.exit(1)
        
    with open(scores_file, "r", encoding="utf-8") as f:
        latest_scores = json.load(f)
        
    faithfulness = latest_scores.get("faithfulness", 0.0)
    relevancy = latest_scores.get("answer_relevancy", 0.0)
    precision = latest_scores.get("context_precision", 0.0)
    
    # 2. Extract commit hash from environment
    commit_sha = os.getenv("GITHUB_SHA", "local-dev")[:7]
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    new_entry = {
        "timestamp": timestamp,
        "commit": commit_sha,
        "faithfulness": round(faithfulness, 2),
        "answer_relevancy": round(relevancy, 2),
        "context_precision": round(precision, 2),
        "hyde_enabled": True
    }
    
    # 3. Read and append to existing history ledger
    history_data = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history_data = json.load(f)
                if not isinstance(history_data, list):
                    history_data = []
        except json.JSONDecodeError:
            history_data = []
            
    history_data.append(new_entry)
    
    # Ensure directory exists and write back
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
        
    print(f"Successfully appended entry for commit {commit_sha} to history.")
    
    # 4. Enforce Quality Gate Threshold
    if faithfulness < 0.05:
        print(f"❌ QUALITY GATE FAILED: Faithfulness ({faithfulness}) is below 0.05.")
        sys.exit(1)
        
    print("✅ QUALITY GATE PASSED.")

if __name__ == "__main__":
    main()