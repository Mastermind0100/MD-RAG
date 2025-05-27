import asyncio
import websockets
import faiss.swigfaiss_avx2
from docCleaner import md_doc_reader
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import numpy as np
import logging
import faiss
import llama_cpp

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
                "filename":"mistral-7b-instruct-v0.2.Q2_K.gguf"
            }
        }

def response_cleaner(og_res:str) -> str:
    new_res = og_res.strip()
    return new_res

def load_llm(model_name: str) -> llama_cpp.llama.Llama:
    parameters = model_parameters[model_name]
    llm = Llama.from_pretrained(repo_id=parameters['repo_id'], filename=parameters['filename'], verbose=False)
    return llm

def load_index(document_embeddings:np.ndarray) -> faiss.swigfaiss_avx2.IndexFlatL2:
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(np.array(document_embeddings))
    return index

def get_context(index:faiss.swigfaiss_avx2.IndexFlatL2, chunkified_document:list[str], query_embeddings:np.ndarray) -> str:
    D, I = index.search(np.array(query_embeddings), k=3)
    top_chunks = [chunkified_document[i] for i in I[0]]
    context = "\n".join(top_chunks)

    return context

document_filepath = "data/README.md"
embedding_model_name = "all-MiniLM-L6-v2"

llm = load_llm("mistral-7b")    

chunkified_document = md_doc_reader(filepath=document_filepath,token_limit=200)
embedder = SentenceTransformer(embedding_model_name)

document_embeddings = embedder.encode(chunkified_document)

index = load_index(document_embeddings)

def get_response(message:str) -> str:
    message_embeddings = embedder.encode([message])
    context = get_context(index, chunkified_document, message_embeddings)
    prompt = ("You are an expert assistant. Using the context provided, give a **concise and factual answer** to the question in plain text. "
                  "Do not use formatting and add any extra content. If you do not find the answer in the context, say you do not know. "
                  f"\nContext: {context}\nQuestion: {message}\nAnswer: ")

    response = llm(prompt, max_tokens=2048)
    cleaned_response = response_cleaner(response['choices'][0]['text'])
    return cleaned_response


async def handler(websocket):
    async for message in websocket:
        print("User:", message)
        response = get_response(message)
        await websocket.send(response)

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Chatbot server running on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
