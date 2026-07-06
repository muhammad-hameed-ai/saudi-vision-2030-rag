import os
import sys
from ingest_data import main as ingest_main
from evaluate_chunking import main as chunk_main

def main():
    print("--- Starting Stage 1: Document Ingestion ---")
    doc_count = ingest_main()
    print(f"--- Stage 1 Complete. Ingested {doc_count} pages. ---\n")

    print("--- Starting Stage 2: Document Chunking ---")
    chunk_count = chunk_main()
    print(f"--- Stage 2 Complete. Created {chunk_count} chunks. ---")

if __name__ == "__main__":
    # Add src directory to path to resolve local imports cleanly
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
