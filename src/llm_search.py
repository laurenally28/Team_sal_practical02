import ollama
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='search.log',
    filemode='a'
)
logger = logging.getLogger(__name__)


def generate_rag_response(query, context_results):
    # Prepare context string
    context_str = "\n".join(
        f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
        f"with similarity {float(result.get('similarity', 0)):.2f}"
        for result in context_results
    )

    logger.debug(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant.
    Use the following context to answer the query as accurately as possible. If the context is
    not relevant to the query, say 'I don't know'.
    Context:
    {context_str}
    Query: {query}
    Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model="llama3.2:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def interactive_search(vector_store):
    """Interactive search interface."""
    logger.info("🔍 RAG Search Interface")
    logger.info("Type 'exit' to quit")
    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = vector_store.search_embeddings(query)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        logger.info("\n--- Response ---")
        logger.info(response)
        print(f"\nResponse: {response}")
