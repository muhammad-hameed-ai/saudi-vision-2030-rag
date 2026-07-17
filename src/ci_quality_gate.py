import json
import os
import sys
from datetime import datetime, timezone
import subprocess

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def main():
    scores_file = "data/evaluation/evaluation_scores.json"
    history_file = "data/evaluation/scores_history.json"
    
    if not os.path.exists(scores_file):
        print(f"Error: {scores_file} not found. Did the evaluation run?")
        sys.exit(1)
        
    with open(scores_file, "r", encoding="utf-8") as f:
        current_scores = json.load(f)
        
    faithfulness = current_scores.get("faithfulness", 0.0)
    relevancy = current_scores.get("answer_relevancy", 0.0)
    precision = current_scores.get("context_precision", 0.0)
    
    commit_hash = get_git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # 1. Update History File
    history = []
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
                
    history.append({
        "timestamp": timestamp,
        "commit": commit_hash,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "precision": precision
    })
    
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    print(f"Scores appended to {history_file}")
    
    # 2. Write to GitHub Actions Step Summary
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    summary_md = (
        "### RAG Evaluation Results\n\n"
        "| Metric | Score |\n"
        "|--------|-------|\n"
        f"| Faithfulness | {faithfulness:.4f} |\n"
        f"| Answer Relevancy | {relevancy:.4f} |\n"
        f"| Context Precision | {precision:.4f} |\n"
    )
    
    if summary_file and os.path.exists(summary_file):
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(summary_md)
    else:
        print("\n--- GitHub Step Summary Preview ---")
        print(summary_md)
        print("-----------------------------------")
        
    # 3. Quality Gate Check
    print(f"\nEvaluating Quality Gate: Faithfulness >= 0.05 (Current: {faithfulness})")
    if faithfulness < 0.05:
        print("❌ QUALITY GATE FAILED: Faithfulness is below 0.05.")
        sys.exit(1)
    else:
        print("✅ QUALITY GATE PASSED.")

if __name__ == "__main__":
    main()