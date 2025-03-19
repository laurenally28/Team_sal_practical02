# laurens first idea
from sentence_transformers import SentenceTransformer

def get_embedding_model(model_name):
    return SentenceTransformer(model_name)

def generate_embeddings(chunks, model):
    return model.encode(chunks)

# Function to compare embedding models
def compare_embedding_models(models, chunks):
    results = {}
    for model_name in models:
        model = get_embedding_model(model_name)
        # Measure speed, memory usage, and qualitative retrieval quality
    return results
