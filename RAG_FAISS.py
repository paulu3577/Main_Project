import os
import re
from uuid import uuid4
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from Drug_details import *
from langchain.docstore.document import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# def create_vector_database():
#     print("Loading PDF...")
#     loader = PyPDFLoader("RAG_FILE.pdf")
#     documents = loader.load()

#     # Create a single text from all documents
#     full_text = "\n".join([doc.page_content for doc in documents])

#     # Split by drug names - this pattern looks for drug names followed by "CIMS Class"
#     drug_sections = re.split(r"([a-zA-Z\-]+)\nCIMS Class", full_text)

#     # Process the sections into documents
#     processed_docs = []
#     for i in range(1, len(drug_sections), 2):  # Skip the first split result and process name+content pairs
#         if i+1 < len(drug_sections):
#             drug_name = drug_sections[i].strip()
#             content = f"{drug_name}\nCIMS Class {drug_sections[i+1]}"

#             doc = Document(
#                 page_content=content,
#                 metadata={"drug_name": drug_name.lower()}
#             )
#             processed_docs.append(doc)

#     print(f"Extracted {len(processed_docs)} drug documents")

#     # Create embeddings
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # Create FAISS vector store
#     vector_store = FAISS.from_documents(processed_docs, embeddings)

#     # Save the vector store locally
#     vector_store.save_local("drug_knowledge_base")

#     print("Drug documents added to FAISS vector store")
#     return vector_store

def query_vector_database(drug_name, top_k=3):
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        # Load the existing vector store
        vector_store = FAISS.load_local("drug_knowledge_base", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return []

    # First try exact match with metadata filter
    
    results = vector_store.similarity_search(
        drug_name, 
        k=top_k, 
        filter={"drug_name": drug_name.lower()}
    )

    # If no results with exact match, try semantic search without filter
    if not results:
        # print(f"No exact match for {drug_name}, trying semantic search...")
        # results = vector_store.similarity_search(drug_name, k=top_k)
        print("From Drugs.com")
        results = drug_details(drug_name)
    return results

# create_vector_database()

# if __name__ == "__main__":
#     # Uncomment to create the database first time
#     # create_vector_database()

#     # Query the database
# results = query_vector_database("abacavir")
# for res in results:
#   print(res)