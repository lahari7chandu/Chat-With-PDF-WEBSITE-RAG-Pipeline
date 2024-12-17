# Chat with PDF Using RAG Pipeline

import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Step 1: Data Ingestion
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def embed_chunks(chunks, model):
    return model.encode(chunks)

def store_embeddings(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Step 2: Query Handling
def query_to_embedding(query, model):
    return model.encode([query])

def retrieve_relevant_chunks(query_embedding, index, chunks, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Step 3: Comparison Queries
def extract_comparison_terms(query):
    # Placeholder for extracting terms from the query
    return query.split(" vs ")

def compare_chunks(chunks, terms):
    # Placeholder for comparison logic
    return f"Comparison results for: {', '.join(terms)}"

# Step 4: Response Generation
def generate_response(retrieved_chunks, query):
    # Placeholder for LLM response generation
    return f"Response based on query: {query} with data: {retrieved_chunks}"

# Example Usage
pdf_path = "path_to_pdf.pdf"
model = SentenceTransformer('all-MiniLM-L6-v2')

text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)
embeddings = embed_chunks(chunks, model)
index = store_embeddings(embeddings)

user_query = "What is the unemployment rate based on degree type?"
query_embedding = query_to_embedding(user_query, model)
relevant_chunks = retrieve_relevant_chunks(query_embedding, index, chunks)

comparison_query = "Compare unemployment rates for Bachelor's vs Master's degrees"
terms = extract_comparison_terms(comparison_query)
comparison_result = compare_chunks(relevant_chunks, terms)

response = generate_response(relevant_chunks, user_query)
print(response)
