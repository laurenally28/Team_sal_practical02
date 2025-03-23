import redis
import numpy as np
from redis.commands.search.query import Query
import logging
import chromadb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ingest.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

class RedisVectorStore:
    def __init__(
        self,
        host="localhost",
        port=6379,
        db=0,
        vector_dim=768,
        index_name="embedding_index",
        doc_prefix="doc:",
        distance_metric="COSINE"
    ):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.vector_dim = vector_dim
        self.index_name = index_name
        self.doc_prefix = doc_prefix
        self.distance_metric = distance_metric
        self.logger = logging.getLogger(__name__)

    def setup_index(self):
        """Clear the Redis store and create the vector index."""
        self.logger.info("Clearing existing Redis store...")
        self.redis_client.flushdb()
        self.logger.info("Redis store cleared.")

        try:
            self.redis_client.execute_command(f"FT.DROPINDEX {self.index_name} DD")
        except redis.exceptions.ResponseError:
            # The index may not exist yet
            pass

        command = f"""
            FT.CREATE {self.index_name} ON HASH PREFIX 1 {self.doc_prefix}
            SCHEMA text TEXT
            embedding VECTOR HNSW 6 DIM {self.vector_dim} TYPE FLOAT32 DISTANCE_METRIC {self.distance_metric}
        """
        self.redis_client.execute_command(command)
        self.logger.info("Index created successfully.")

    def store_embedding(self, file: str, page: str, chunk: str, embedding: list):
        """Store the embedding in Redis with associated metadata."""
        key = f"{self.doc_prefix}:{file}_page_{page}_chunk_{chunk}"
        self.redis_client.hset(
            key,
            mapping={
                "file": file,
                "page": page,
                "chunk": chunk,
                "embedding": np.array(embedding, dtype=np.float32).tobytes(),
            },
        )
        self.logger.debug(f"Stored embedding for chunk: {chunk}")
    def search_embeddings(self, query_embedding, top_k=3):
        #query_embedding = get_embedding(query)
        # Convert embedding to bytes for Redis search
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        try:
            # Construct the vector similarity search query
            # Use a more standard RediSearch vector search syntax
            # q = Query("*").sort_by("embedding", query_vector)
            q = (
                Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
                .sort_by("vector_distance")
                .return_fields("id", "file", "page", "chunk", "vector_distance")
                .dialect(2)
            )

            # Perform the search
            results = self.redis_client.ft(self.index_name).search(
                q, query_params={"vec": query_vector}
            )

            # Transform results into the expected format
            top_results = [
                {
                    "file": result.file,
                    "page": result.page,
                    "chunk": result.chunk,
                    "similarity": result.vector_distance,
                }
                for result in results.docs
            ][:top_k]

            # Print results for debugging
            for result in top_results:
                logger.debug(
                    f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
                )

            return top_results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []


class ChromaVectorStore:
    def __init__(self, collection_name="ds_4300_proj_2", embedding_function=None):
        """
        Initialize the ChromaDB client and collection.
        
        Parameters:
            collection_name (str): Name of the collection.
            embedding_function (callable, optional): If provided, the collection will use this function
                to compute embeddings when adding documents without a precomputed embedding.
        """
        self.client = chromadb.HttpClient(host="localhost", port=8000)
        if collection_name in self.client.list_collections():
            self.collection = self.client.get_collection(name=collection_name, embedding_function=embedding_function)
        else:
            self.collection = self.client.create_collection(name=collection_name, embedding_function=embedding_function)

        logger.info(f"ChromaDBStore collection '{collection_name}' initialized.")

    def setup_index(self):
        """
        For ChromaDB, you might want to clear an existing collection or recreate it.
        Here we'll assume that creating the collection is sufficient.
        """
        logger.info("ChromaDBStore setup_index: collection is ready for use.")

    def store_embedding(self, file: str, page: str, chunk: str, embedding: list):
        """
        Store a document (or text chunk) along with its embedding and metadata"""
        # Create a unique id (here using file, page, and a hash of the chunk)
        doc_id = f"{file}_page_{page}_chunk_{abs(hash(chunk))}"
        metadata = {"file": file, "page": page, "chunk": chunk}
        self.collection.add(
            documents=[chunk],
            metadatas=[metadata],
            ids=[doc_id],
            embeddings=[embedding]  # supplying our own embedding
        )
        logger.debug(f"ChromaDBStore: Stored embedding for {doc_id}")

    def search_embeddings(self, query_embedding, top_k=3):
        """
        Query the collection for documents similar to the given query embedding.
        """
        # Perform the query using the query embedding.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        formatted_results = []
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc_id, meta, distance in zip(ids, metadatas, distances):
            formatted_results.append({
                "file": meta.get("file", ""),
                "page": meta.get("page", ""),
                "chunk": meta.get("chunk", ""),
                "similarity": distance,
            })
        logger.debug("ChromaDBStore: search_embeddings completed.")
        return formatted_results




def get_vector_store(db_name: str):
    """Return the desired vector store based on configuration."""
    if db_name == "redis":
        return RedisVectorStore()
    elif db_name == "chroma":
        return ChromaVectorStore()
    else:
        raise ValueError(f"Unsupported vector database: {db_name}")