import os
import pickle
import json
import re
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def detect_section_header(line):
    """Detect if a line is a section header based on structural cues."""
    stripped = line.strip()
    if not stripped:
        return None
    # All-caps lines under 120 chars are likely headers
    if stripped.isupper() and len(stripped) < 120 and len(stripped) > 3:
        return stripped
    # Lines starting with numbered patterns like "1.", "1.1", "Chapter 1"
    if re.match(r'^(?:Chapter|Section|Part)\s+\d', stripped, re.IGNORECASE):
        return stripped
    if re.match(r'^\d+(?:\.\d+)*\s+[A-Z]', stripped):
        return stripped
    return None


def clean_and_tag_text(text):
    """Apply structural tagging to raw PDF text for header-aware splitting."""
    table_pattern = re.compile(
        r'(?:\|\s*.+\s*\|)+|(?:\w+\s{2,}\d+[\d\s,.]*)'
    )
    lines = text.split("\n")
    processed = []
    for line in lines:
        if table_pattern.match(line.strip()):
            processed.append(f"[TABLE_ROW] {line}")
        elif detect_section_header(line):
            processed.append(f"[SECTION_HEADER] {line}")
        else:
            processed.append(line)
    return "\n".join(processed)


def strip_structural_markers(text):
    """Remove [TABLE_ROW] and [SECTION_HEADER] prefixes from final chunk text."""
    text = re.sub(r'\[TABLE_ROW\]\s*', '', text)
    text = re.sub(r'\[SECTION_HEADER\]\s*', '', text)
    return text.strip()


def extract_section_from_chunk(text):
    """Find the nearest [SECTION_HEADER] in the chunk text to use as metadata."""
    match = re.search(r'\[SECTION_HEADER\]\s*(.+)', text)
    if match:
        return match.group(1).strip()
    return "General"


def main():
    params = load_params()
    ingest_cfg = params["ingest"]
    chunk_cfg = params["chunk"]

    print(f"Loading documents from {ingest_cfg['output_path']}...")
    with open(ingest_cfg["output_path"], "rb") as f:
        documents = pickle.load(f)
    print(f"Loaded {len(documents)} pages")

    print("Applying structural tagging...")
    for doc in documents:
        doc.page_content = clean_and_tag_text(doc.page_content)

    # Use section headers as primary split boundaries, then fall back to
    # paragraph and sentence boundaries within each section.
    separators = [
        "\n[SECTION_HEADER]",  # Split on section headers first
        "\n\n",                # Then paragraph breaks
        "\n",                  # Then line breaks
        ". ",                  # Then sentence boundaries
        " ",                   # Then words
        ""
    ]

    print(f"Chunking: size={chunk_cfg['chunk_size']}, overlap={chunk_cfg['chunk_overlap']}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"],
        separators=separators
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Enrich metadata and clean marker prefixes from final text
    print("Enriching chunk metadata and cleaning markers...")
    for chunk in chunks:
        # Extract section name before stripping markers
        section = extract_section_from_chunk(chunk.page_content)
        chunk.metadata["section"] = section

        # Strip the structural markers from the text that will be embedded
        chunk.page_content = strip_structural_markers(chunk.page_content)

    # Filter out empty chunks
    chunks = [c for c in chunks if len(c.page_content.strip()) > 20]
    print(f"After filtering: {len(chunks)} chunks")

    os.makedirs(os.path.dirname(chunk_cfg["output_path"]), exist_ok=True)
    with open(chunk_cfg["output_path"], "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved chunks to {chunk_cfg['output_path']}")

    metrics_path = "data/processed_data/chunk_metrics.json"
    metrics = {
        "total_chunks": len(chunks),
        "avg_chunk_length": round(
            sum(len(c.page_content) for c in chunks) / len(chunks), 2
        ),
        "chunk_size": chunk_cfg["chunk_size"],
        "chunk_overlap": chunk_cfg["chunk_overlap"],
        "chunking_strategy": "structure_aware_recursive",
        "separators": separators,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics: {metrics}")
    return len(chunks)


if __name__ == "__main__":
    main()