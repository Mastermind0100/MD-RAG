import faiss
from docCleaner import md_doc_reader
from sentence_transformers import SentenceTransformer
import json
import numpy as np

def main():
    chunkified_document = md_doc_reader('data/README.md', 120)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    document_embeddings = embedder.encode(chunkified_document)

    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(np.array(document_embeddings))

    faiss.write_index(index, "data/faiss_index.idx")

    with open('data/document_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunkified_document, f, indent=4)

if __name__ == "__main__":
    main()