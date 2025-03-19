from ingest import load_documents, preprocess_text, chunk_text
from embedding import get_embedding_model, generate_embeddings, compare_embedding_models
from vector_db import initialize_vector_db, index_documents
from llm import initialize_llm, compare_llm_models
from query import process_query

def run_experiment(chunk_size, overlap, embedding_model, vector_db, llm_model):
    # Load and preprocess documents
    documents = load_documents('data/')
    preprocessed_docs = [preprocess_text(doc) for doc in documents]
    chunks = [chunk_text(doc, chunk_size, overlap) for doc in preprocessed_docs]

    # Generate embeddings
    embedding_model = get_embedding_model(embedding_model)
    embeddings = generate_embeddings(chunks, embedding_model)

    # Index documents
    db = initialize_vector_db(vector_db)
    index_documents(db, embeddings, metadata)

    # Initialize LLM
    llm = initialize_llm(llm_model)

    # Run queries and collect metrics
    results = run_queries(db, embedding_model, llm)
    return results

def run_queries(db, embedding_model, llm):
    test_queries = [
        "What is the main topic of DS4300?",
        "Explain the concept of vector databases.",
        # Add more test queries
    ]
    results = []
    for query in test_queries:
        response = process_query(query, db, embedding_model, llm)
        results.append((query, response))
    return results

if __name__ == "__main__":
    # Run experiments with different configurations
    configurations = [
        (500, 50, 'sentence-transformers/all-MiniLM-L6-v2', 'redis', 'llama2:7b'),
        (1000, 100, 'sentence-transformers/all-mpnet-base-v2', 'chroma', 'mistral:7b'),
        # Add more configurations
    ]
    
    for config in configurations:
        results = run_experiment(*config)
        # Analyze and store results
    
    # Compare embedding models
    embedding_models = ['sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/all-mpnet-base-v2', 'InstructorXL']
    embedding_comparison = compare_embedding_models(embedding_models, chunks)
    
    # Compare LLM models
    llm_models = ['llama2:7b', 'mistral:7b']
    llm_comparison = compare_llm_models(llm_models, test_prompts)
    
    # Analyze results and prepare final report
