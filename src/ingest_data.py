import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

RAW_DATA_DIR = "data/raw_pdfs"
PROCESSED_DATA_DIR = "data/processed_data"

def main():
    print(f"Starting ingestion of 48 PDFs from {RAW_DATA_DIR}...")

    # 1. Load all PDFs from the directory
    loader = PyPDFDirectoryLoader(RAW_DATA_DIR)
    documents = loader.load()

    print(f"Successfully extracted {len(documents)} total pages from your files.")

    # 2. Chunk the text split down by paragraphs, sentences, then spaces
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} text chunks for the Vector Database.")

    # 3. Save chunks to data/processed_data/
    save_path = os.path.join(PROCESSED_DATA_DIR, "document_chunks.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Chunks safely stored in {save_path}!")

if __name__ == "__main__":
    main()