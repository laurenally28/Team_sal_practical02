from embedding import get_embedding_model, generate_embeddings
from vector_db import query_vector_db
from llm import generate_response

def process_query(query, vector_db, embedding_model, llm):
    query_embedding = generate_embeddings([query], embedding_model)[0]
    relevant_docs = query_vector_db(vector_db, query_embedding)
    context = prepare_context(relevant_docs)
    prompt = construct_prompt(query, context)
    response = generate_response(llm, prompt)
    return response

def prepare_context(relevant_docs):
    # Prepare context from relevant documents
    pass

def construct_prompt(query, context):
    # Construct prompt using query and context
    pass
