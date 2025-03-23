import os 
import time
import logging
import csv
import time
import psutil  # optional, if you want to track memory usage
import pandas as pd 
from src.preprocess_text import extract_text_from_pdf, preprocess_text, split_text_into_chunks
from src.embeddings import get_embedding
from src.vector_store import get_vector_store
from src.llm_search import generate_rag_response


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pdfs(data_dir, embedding_model, vector_store, chunk_size, overlap, prep_strategy):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                processed_text = preprocess_text(text, strategy=prep_strategy)
                chunks = split_text_into_chunks(processed_text, chunk_size, overlap)
                
                # print(f" Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(processed_text, embedding_model)
                    vector_store.store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
 
                    #print(f" -----> Processed {file_name}")
def interactive_search(vector_store, initial_query=None):
    """Interactive search interface."""
    logger.info("🔍 RAG Search Interface")
    logger.info("Type 'exit' to quit")
    query = initial_query
    results = []
    while True:
        start_time = time.time()
        if not query:
            query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        # Get query embedding
        query_embedding = get_embedding(query)

        # Search for relevant embeddings
        context_results = vector_store.search_embeddings(query_embedding)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print(f"\nResponse: {response}")
        logger.info("--- End of Response ---")

        # Reset query so user is prompted again
        
        end = time.time()
        results.append({
            "query": query,
            "rag_response": response,
            "response_time": end - start_time
        })
        query = None

    return results
    
        

 # optional alternative using pandas

def save_experiment_results(results, csv_file_path):
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file_path)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Open file in append mode
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(results[0].keys()) if results else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write header only if the file didn't exist before
        if not file_exists:
            writer.writeheader()
        for row in results:
            writer.writerow(row)

def main():
    data_dir = "data"
    embedding_model_name = "nomic-embed-text"  # or your preferred model name
    vector_db_name = "chroma"  # or "chroma"
    chunk_size = 300
    overlap = 50
    prep_strategy = 'basic'
    start_time = time.time()
    vector_store = get_vector_store(vector_db_name)
    vector_store.setup_index()

    # Record start time and optional memory usage
    
    process_pdfs(data_dir, embedding_model_name, vector_store, chunk_size, overlap, prep_strategy)

    query = "What is an AVL Tree?"
    results = interactive_search(vector_store, initial_query=query)

    end_time = time.time()
    total_pipeline_time = end_time - start_time
    mem_used = psutil.Process().memory_info().rss  # memory usage in bytes
     # response from the RAG model

     # Combine experiment variables with each query result into one list of dictionaries
    experiment_results = []
    for res in results:
        row = {
            'embedding_model_name': embedding_model_name,  # or your preferred model name
            'vector_db_name': vector_db_name,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "prep_strategy": prep_strategy,
            "response_time": res['response_time'],
            "memory_usage_bytes": mem_used,
            "query": res["query"],
            "rag_response": res["rag_response"],
            "total_pipeline_time": total_pipeline_time,
        }
        experiment_results.append(row)

    # Save the combined experiment results to a CSV file
    csv_file_path = "results/experiment_results.csv"
    save_experiment_results(experiment_results, csv_file_path)
    print("saved!")

if __name__ == "__main__":
    main()


