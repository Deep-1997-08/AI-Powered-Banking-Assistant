from langchain import LangChain
from haystack import Finder
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
from haystack.reader.farm import FARMReader

def search_documents(query):
    document_store = InMemoryDocumentStore()
    retriever = TfidfRetriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

    finder = Finder(reader, retriever)
    documents = document_store.get_all_documents()
    results = finder.get_answers(question=query, documents=documents)
    return results
