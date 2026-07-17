with open(r".github\workflows\pipeline.yml", "r", encoding="utf-8") as f:
    content = f.read()

new_step = """
      - name: Run Quality Gate Check
        run: python src/ci_quality_gate.py
"""

if "Run Quality Gate Check" not in content:
    content += new_step
    with open(r".github\workflows\pipeline.yml", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: pipeline.yml has been updated with the Quality Gate step.")
else:
    print("WARNING: Quality Gate step is already in the pipeline.")