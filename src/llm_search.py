import ollama
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="search.log",
    filemode="a",
)
logger = logging.getLogger(__name__)

# Available models
AVAILABLE_MODELS = {
    "llama": "llama3.2:latest",
    "mistral": "mistral:latest",
}

def generate_rag_response(query, context_results, model_type="llama"):
    """Generate a RAG response using a specified local model."""
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"Invalid model type. Choose from {list(AVAILABLE_MODELS.keys())}")

    context_str = "\n".join(
        f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
        f"with similarity {float(result.get('similarity', 0)):.2f}"
        for result in context_results
    )

    logger.debug(f"context_str: {context_str}")

    prompt = f"""You are a helpful AI assistant.
    Use the following context to answer the query as accurately as possible. If the context is
    not relevant to the query, say 'I don't know'.
    Context:
    {context_str}
    Query: {query}
    Answer:"""

    # Generate response using Ollama with the selected model
    response = ollama.chat(
        model=AVAILABLE_MODELS[model_type], messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def interactive_search(vector_store, model_type="llama"):
    """Interactive search interface allowing model selection."""
    logger.info(f"🔍 RAG Search Interface using {model_type.upper()}")
    logger.info("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = vector_store.search_embeddings(query)

        # Generate RAG response using the selected model
        response = generate_rag_response(query, context_results, model_type)

        logger.info("\n--- Response ---")
        logger.info(response)
        print(f"\nResponse: {response}")

