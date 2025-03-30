# Team_sal_practical02
In this project, you and your team will build a local Retrieval-Augmented Generation system that allows a user to query the collective DS4300 notes from members of your team. 

## Source Code
- `pipeline.py` - Processes PDFs by extracting, preprocessing, and embedding text into a vector database for retrieval-augmented generation (RAG)-based search. It runs experiments with different configurations, benchmarks query performance, and logs results, making it useful for evaluating embeddings and vector database performance.
- `src/embeddings.py` - Generates text embeddings using either Sentence-Transformers or Ollama's nomic-embed-text model. It dynamically selects the embedding method based on the specified model, allowing flexibility in embedding generation for downstream tasks.
- `src/vector_store.py` - 
- `src/llm_search.py` - 
- `src/preprocess_text.py` -  


