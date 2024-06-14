import pinecone
from src.config import PINECONE_API_KEY

def setup_vector_db(documents):
    pinecone.init(api_key=PINECONE_API_KEY, environment='us-west1-gcp')
    index = pinecone.Index("banking-documents")
    for doc in documents:
        index.upsert([(doc['doc_id'], doc['doc_text'])])
    return index
