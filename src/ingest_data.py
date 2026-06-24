import os
import pickle
import yaml
from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    cfg = params["ingest"]

    print(f"Loading PDFs from {cfg['data_dir']}...")
    loader = PyPDFDirectoryLoader(cfg["data_dir"])
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF corpus")

    os.makedirs(os.path.dirname(cfg["output_path"]), exist_ok=True)
    with open(cfg["output_path"], "wb") as f:
        pickle.dump(documents, f)

    print(f"Saved {len(documents)} documents to {cfg['output_path']}")
    return len(documents)


if __name__ == "__main__":
    main()