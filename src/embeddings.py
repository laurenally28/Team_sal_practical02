import time
import tracemalloc
import logging
from sentence_transformers import SentenceTransformer
import ollama
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    logger.info(f"Getting embedding for model: {model}")
    
    if model == "nomic-embed-text": 
        embeddings = ollama_models(text, model)
    elif model == "distilroberta-base":
        embeddings = distilroberta_base_embedding(text)
    else: 
        embeddings = sent_transformer_models(text, model)

    return embeddings

def sent_transformer_models(text: str, model_name: str): 
    model = SentenceTransformer(model_name, device="cpu")  # Force CPU usage
    return model.encode(text).tolist()


def distilroberta_base_embedding(text: str):
    """Manually extract embeddings from distilroberta-base using mean pooling."""
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state  # Get hidden states

    # Mean pooling across token embeddings to get a single 768D vector
    return outputs.mean(dim=1).squeeze().tolist()

def ollama_models(text: str, model_name: str): 
    response = ollama.embeddings(model=model_name, prompt=text)
    return response["embedding"]

