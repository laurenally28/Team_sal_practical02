import os
import time
import logging
import csv
import argparse
import psutil
import pandas as pd
import itertools

from src.preprocess_text import extract_text_from_pdf, preprocess_text, split_text_into_chunks
from src.embeddings import get_embedding
from src.vector_store import get_vector_store
from src.llm_search import generate_rag_response, interactive_search

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

                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, embedding_model)
                    print(len(embedding))
                    vector_store.store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )


def interactive_search(vector_store, initial_query=None):
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

        query_embedding = get_embedding(query)
        context_results = vector_store.search_embeddings(query_embedding)
        response = generate_rag_response(query, context_results)

        print(f"\nResponse: {response}")
        logger.info("--- End of Response ---")

        end_time = time.time()
        results.append({
            "query": query,
            "rag_response": response,
            "response_time": end_time - start_time
        })
        query = None
    return results

def run_predefined_search(vector_store, queries):
    """Run a predefined list of queries for benchmarking"""
    logger.info("Running predefined queries...")
    results = []

    for query in queries:
        logger.info(f"Query: {query}")
        start_time = time.time()

        query_embedding = get_embedding(query)
        context_results = vector_store.search_embeddings(query_embedding)
        response = generate_rag_response(query, context_results)

        end_time = time.time()
        results.append({
            "query": query,
            "rag_response": response,
            "response_time": end_time - start_time
        })

        logger.info(f"Response: {response}")
        logger.info("--- End of Query ---")

    return results



def save_experiment_results(results, csv_file_path):
    file_exists = os.path.isfile(csv_file_path)
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(results[0].keys()) if results else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in results:
            writer.writerow(row)


def run_experiment(config, query, mode):
    data_dir = "data"
    start_time = time.time()

    vector_store = get_vector_store(config["vector_db"])
    vector_store.setup_index()

    mem_before = psutil.Process().memory_info().rss

    process_pdfs(
        data_dir,
        config["embedding_model"],
        vector_store,
        config["chunk_size"],
        config["overlap"],
        config["prep_strategy"]
    )
    if mode == "interactive":
        results = interactive_search(vector_store, initial_query=query) 
    else:
        results = run_predefined_search(vector_store, queries=query)

    mem_after = psutil.Process().memory_info().rss
    total_pipeline_time = time.time() - start_time

    experiment_results = []
    for res in results:
        row = {
            **config,
            "query": res["query"],
            "rag_response": res["rag_response"],
            "response_time": res["response_time"],
            "memory_usage_bytes": mem_after - mem_before,
            "total_pipeline_time": total_pipeline_time,
        }
        experiment_results.append(row)

    return experiment_results


def main():
    # Parameter spaces
    # need 786 dimensional embeddings
    args = get_cli_args()
    if args.mode == "interactive":
        config = {
                "embedding_model": "nomic-embed-text",
                "vector_db": "redis",
                "chunk_size": 300,
                "overlap": 50,
                "prep_strategy": "basic"
            }
        run_experiment(config, query=None, mode=args.mode) 
        
    else:
        embedding_models = ["nomic-embed-text", "sentence-transformers/paraphrase-albert-small-v2", "distilroberta-base"]
        vector_dbs = ["faiss"]
        #vector_dbs = ["redis", "chroma", "faiss"]
        chunk_sizes = [300, 500]
        overlaps = [0, 50]
        prep_strategies = ["basic"]

        # Static query for testing
        query = ["What is an AVL Tree?", "What is the difference between a list where memory is contiguously allocated and a list where linked structures are used?",
                "When are linked lists faster than contiguously-allocated lists?"]

        csv_file_path = "results/experiment_results.csv"

        # Generate all combinations
        all_combinations = list(itertools.product(
            embedding_models,
            vector_dbs,
            chunk_sizes,
            overlaps,
            prep_strategies
        ))

        logger.info(f"Running {len(all_combinations)} configurations...")

        for embedding_model, vector_db, chunk_size, overlap, prep_strategy in all_combinations:
            config = {
                "embedding_model": embedding_model,
                "vector_db": vector_db,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "prep_strategy": prep_strategy
            }

            logger.info(f"Running config: {config}")
            experiment_results = run_experiment(config, query, mode=args.mode)
            save_experiment_results(experiment_results, csv_file_path)

        logger.info("All experiments completed!")
    

def get_cli_args(): 
    parser = argparse.ArgumentParser(description="pipeline execution for exam RAG")

    # Add arguments
    parser.add_argument('--mode', type=str, help='specifies if pipeline is interactive (exam ver). No arg assumes full experiment execution.', default='experiments')

    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
