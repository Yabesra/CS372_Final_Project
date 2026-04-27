# src/retriever.py
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)


def embed(texts):
    """
    function: converts text into dense vectors via mean 
    pooling over last_hidden_state.
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def retrieve(query, chunks, chunk_embeddings, top_k=1):
    """
    function: computes cosine similarity between the query 
    embedding and all chunk embeddings
    """
    query_emb = embed([query])[0]

    sims = np.dot(chunk_embeddings, query_emb) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb)
    )

    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in top_idx]