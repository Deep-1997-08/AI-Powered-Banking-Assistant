from src import (
    speech_to_text,
    nlp_model,
    semantic_search,
    agentic_workflows,
    vector_database,
    embedding_creation
)
from src.utils import load_data, preprocess_queries

def main():
    customer_queries = load_data("data/customer_queries.csv")
    banking_documents = load_data("data/banking_documents.csv")

    # Preprocess queries
    query_texts = preprocess_queries(customer_queries['query_text'])

    # Speech to Text (Assume we have audio files for each query)
    transcribed_queries = [speech_to_text.transcribe(f"data/audio/query_{i}.wav") for i in range(len(query_texts))]

    # NLP Model
    responses = [nlp_model.generate_response(query) for query in transcribed_queries]

    # Semantic Search
    search_results = [semantic_search.search_documents(query) for query in transcribed_queries]

    # Agentic Workflows
    workflow_results = [agentic_workflows.handle_query(query) for query in transcribed_queries]

    # Vector Database
    vector_db = vector_database.setup_vector_db(banking_documents)

    # Embedding Creation
    embeddings = embedding_creation.create_embeddings(banking_documents['doc_text'])

    # Example output
    print("Responses:", responses)
    print("Search Results:", search_results)
    print("Workflow Results:", workflow_results)

if __name__ == "__main__":
    main()
