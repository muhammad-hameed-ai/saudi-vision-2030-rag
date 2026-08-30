with open(r"src\evaluate_rag.py", "r", encoding="utf-8") as f:
    content = f.read()

old_str = 'url="http://localhost:6333",'
new_str = 'url="http://localhost:6333",\n        vector_name="dense",'

if 'vector_name="dense"' not in content:
    content = content.replace(old_str, new_str)
    with open(r"src\evaluate_rag.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: src\\evaluate_rag.py has been updated to use the 'dense' vector name.")
else:
    print("WARNING: 'dense' vector name is already in the file.")