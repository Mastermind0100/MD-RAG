import faiss.swigfaiss_avx2
from docCleaner import md_doc_reader
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import numpy as np
import logging
import faiss
import llama_cpp
import json

logging.getLogger("llama_cpp").setLevel(logging.WARNING)

model_parameters = {
            "medQwen": {
                "repo_id": "mradermacher/Qwen-3-32B-Medical-Reasoning-i1-GGUF",
                "filename": "Qwen-3-32B-Medical-Reasoning.i1-IQ1_M.gguf"
            },
            "gemma3-1b": {
                "repo_id": "google/gemma-3-1b-it-qat-q4_0-gguf",
                "filename": "gemma-3-1b-it-q4_0.gguf"
            },
            "gemma-2b": {
                "repo_id": "google/gemma-2b-it-GGUF",
                "filename": "gemma-2b-it.gguf"
            },
            "mistral-7b": {
                "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "filename":"mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            },
            "medgamma": {
                "repo_id": "gaianet/medgemma-4b-it-GGUF",
                "filename":"medgemma-4b-it-Q2_K.gguf"
            }
        }

def response_cleaner(og_res:str) -> str:
    new_res = og_res.strip()
    return new_res

def load_llm(model_name: str) -> llama_cpp.llama.Llama:
    parameters = model_parameters[model_name]
    llm = Llama.from_pretrained(
        repo_id=parameters['repo_id'], 
        filename=parameters['filename'], 
        verbose=False, 
        n_ctx = 32768,
        n_gpu_layers = 30
    )
    return llm

def load_index(index_filepath:str) -> faiss.swigfaiss_avx2.IndexFlatL2:
    index = faiss.read_index(index_filepath)
    return index

def load_document(document_filepath:str) -> list[str]:
    with open(document_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_context(index:faiss.swigfaiss_avx2.IndexFlatL2, chunkified_document:list[str], query_embeddings:np.ndarray) -> str:
    D, I = index.search(np.array(query_embeddings), k=3)
    top_chunks = [chunkified_document[i] for i in I[0]]
    context = "\n".join(top_chunks)

    return context

class RAGbot:
    def __init__(self, document_filepath:str = "data/document_chunks.json", embedding_model_name:str = "all-MiniLM-L6-v2", 
                 index_filepath:str = "data/faiss_index.idx", model_name:str = "mistral-7b"):
        self.document = load_document(document_filepath)
        self.embedder = SentenceTransformer(embedding_model_name)
        self.index = load_index(index_filepath)
        self.llm = load_llm(model_name)

    def chat(self, query:str) -> str:
        query_embeddings = self.embedder.encode([query])
        context = get_context(self.index, self.document, query_embeddings)
        prompt = ("You are an expert virtual assistant for the PROTON Website. Using the context provided, give a **short and factual answer** to the question in plain text. "
                  "Do not use formatting and add any extra content. If you are asked a question that is not relevant to the context, reply: 'I am sorry, I cannot answer this question.'. "
                  "If the question is relevant to the context but not explicitly mentioned, reply: 'Please reach out to contact@protonstudy.com for this query!'"
                  "If the user uses pleasantries, reply in kind but quickly ask them to stick to PROTON related questions."
                  f"\nContext: {context}\nQuestion: {query}\nAnswer: ")
        response = self.llm(prompt, max_tokens=4096)
        cleaned_response = response_cleaner(response['choices'][0]['text'])
        return cleaned_response

def main():
    document_filepath = "data/document_chunks.json"
    embedding_model_name = "all-MiniLM-L6-v2"
    index_filepath = "data/faiss_index.idx"

    llm = load_llm("mistral-7b")    
    
    embedder = SentenceTransformer(embedding_model_name)

    index = load_index(index_filepath)
    chunkified_document = load_document(document_filepath)

    exitflag = 0

    while True:
        query = input("\nEnter your Question: ")
        if query.lower().strip() == "exit":
            exitflag = 1
            break
        query_embeddings = embedder.encode([query])

        context = get_context(index, chunkified_document, query_embeddings)
        # print(context)

        prompt = ("You are an expert assistant. Using the context provided, give a **short and factual answer** to the question in plain text. "
                  "Do not use formatting and add any extra content. If you are asked a question that is not relevant to the context, reply: 'I am sorry, I cannot answer this question.'. "
                  "If the question is relevant to the context but not explicitly mentioned, reply: 'Please reach out to Rachel for this query!'"
                  f"\nContext: {context}\nQuestion: {query}\nAnswer: ")

        response = llm(prompt, max_tokens=4096)
        cleaned_response = response_cleaner(response['choices'][0]['text'])

        print(cleaned_response)
    
    if exitflag == 1:
        print("Thank you and have a nice day!")

if __name__ == "__main__":
    main()
