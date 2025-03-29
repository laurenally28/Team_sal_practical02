import time
import tracemalloc
import logging
from sentence_transformers import SentenceTransformer
import ollama

logger = logging.getLogger(__name__)

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    logger.info(f" Getting embedding for model: {model}")
    
    if model == "nomic-embed-text": 
        embeddings = ollama_models(text, model)
    else: 
        embeddings = sent_transformer_models(text, model)


    return embeddings

def sent_transformer_models(text: str, model_name: str): 
    model = SentenceTransformer(model_name)
    return model.encode(text).tolist()

def ollama_models(text: str, model_name: str): 
    response = ollama.embeddings(model=model_name, prompt=text)
    return response["embedding"]

