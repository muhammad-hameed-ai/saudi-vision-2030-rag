import os
import pickle
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data/raw_pdfs"
OUTPUT_DIR = "data/processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_and_tag_text(text):
    """
    Preprocesses text to identify structural elements like tables and headers
    before chunking occurs.
    """
    # Regex to capture possible table rows (e.g., numbers separated by multiple spaces or pipes)
    table_pattern = re.compile(r'(?:\|\s*.+\s*\|)+|(?:\w+\s{2,}\d+[\d\s,.]*)')
    
    # Tag potential structural blocks to preserve context
    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        if table_pattern.match(line.strip()):
            processed_lines.append(f"[TABLE_ROW] {line}")
        elif line.strip().isupper() and len(line.strip()) < 100:
            processed_lines.append(f"[SECTION_HEADER] {line}")
        else:
            processed_lines.append(line)
            
    return "\n".join(processed_lines)

def main():
    print("Executing Day 2 Advanced Pipeline...")
    
    # 1. Load Documents
    print("Loading PDFs from directory...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} total pages.")
    
    # Apply structural tagging preprocessing
    print("Preprocessing text for table and header retention...")
    for doc in raw_documents:
        doc.page_content = clean_and_tag_text(doc.page_content)

    # 2. Strategy A: Fixed-Size Chunking (Baseline)
    print("\n--- Running Strategy A: Fixed-Size Chunking ---")
    fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    fixed_chunks = fixed_splitter.split_documents(raw_documents)
    print(f"Fixed Chunking Total: {len(fixed_chunks)} chunks.")

    # 3. Strategy B: Recursive Character Splitting (With Overlap)
    print("\n--- Running Strategy B: Recursive Character Splitting ---")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    recursive_chunks = recursive_splitter.split_documents(raw_documents)
    print(f"Recursive Chunking Total: {len(recursive_chunks)} chunks.")

    # 4. Strategy C: Semantic Chunking (Meaning-Driven)
    print("\n--- Running Strategy C: Semantic Chunking ---")
    print("Initializing embedding engine for sentence breakdown...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    semantic_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile" # Splits based on statistical differences between sentences
    )
    semantic_chunks = semantic_splitter.split_documents(raw_documents)
    print(f"Semantic Chunking Total: {len(semantic_chunks)} chunks.")

    # 5. Export Strategy B as our high-fidelity baseline production choice
    production_path = os.path.join(OUTPUT_DIR, "document_chunks.pkl")
    with open(production_path, "wb") as f:
        pickle.dump(recursive_chunks, f)
        
    print(f"\n✅ Pipeline complete. Production choice saved to {production_path}")

if __name__ == "__main__":
    main()