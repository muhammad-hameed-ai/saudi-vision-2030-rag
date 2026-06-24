import os
import pickle
import yaml
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def clean_and_tag_text(text):
    table_pattern = re.compile(
        r'(?:\|\s*.+\s*\|)+|(?:\w+\s{2,}\d+[\d\s,.]*)'
    )
    lines = text.split("\n")
    processed = []
    for line in lines:
        if table_pattern.match(line.strip()):
            processed.append(f"[TABLE_ROW] {line}")
        elif line.strip().isupper() and len(line.strip()) < 100:
            processed.append(f"[SECTION_HEADER] {line}")
        else:
            processed.append(line)
    return "\n".join(processed)


def main():
    params = load_params()
    ingest_cfg = params["ingest"]
    chunk_cfg  = params["chunk"]

    print(f"Loading documents from {ingest_cfg['output_path']}...")
    with open(ingest_cfg["output_path"], "rb") as f:
        documents = pickle.load(f)
    print(f"Loaded {len(documents)} pages")

    print("Applying structural tagging...")
    for doc in documents:
        doc.page_content = clean_and_tag_text(doc.page_content)

    print(f"Chunking: size={chunk_cfg['chunk_size']}, overlap={chunk_cfg['chunk_overlap']}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    os.makedirs(os.path.dirname(chunk_cfg["output_path"]), exist_ok=True)
    with open(chunk_cfg["output_path"], "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved chunks to {chunk_cfg['output_path']}")

    metrics_path = "data/processed_data/chunk_metrics.json"
    import json
    metrics = {
        "total_chunks": len(chunks),
        "avg_chunk_length": round(
            sum(len(c.page_content) for c in chunks) / len(chunks), 2
        ),
        "chunk_size": chunk_cfg["chunk_size"],
        "chunk_overlap": chunk_cfg["chunk_overlap"],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {metrics}")
    return len(chunks)


if __name__ == "__main__":
    main()