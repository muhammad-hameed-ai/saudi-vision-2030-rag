import subprocess
import os
import sys

def run_eval(use_hyde_flag):
    env = os.environ.copy()
    env["USE_HYDE"] = use_hyde_flag
    # Force Python to use the correct IPv4/IPv6 loopback for Windows
    env["OLLAMA_HOST"] = "http://localhost:11434"
    print(f"\n--- Running Evaluation with USE_HYDE={use_hyde_flag} ---")
    
    result = subprocess.run(
        [sys.executable, "src\\evaluate_rag.py"],
        env=env,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:")
        print(result.stderr)

if __name__ == "__main__":
    # 1. Run Baseline (No HyDE)
    run_eval("false")
    
    # 2. Run with HyDE Enabled
    run_eval("true")