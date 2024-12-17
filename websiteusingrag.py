# Chat with Website Using RAG Pipeline

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class RAGPipeline:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model_name)
        self.vector_database = []
        self.metadata = []

    def crawl_and_scrape(self, urls):
        for url in urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text()
            self.process_content(text_content, url)

    def process_content(self, content, url):
        chunks = self.segment_content(content)
        embeddings = self.model.encode(chunks)
        self.store_embeddings(embeddings, url)

    def segment_content(self, content):
        return content.split('\n\n')  # Simple segmentation by paragraphs

    def store_embeddings(self, embeddings, url):
        for embedding in embeddings:
            self.vector_database.append(embedding)
            self.metadata.append(url)

    def query(self, user_query):
        query_embedding = self.model.encode([user_query])
        distances, indices = self.similarity_search(query_embedding)
        return self.generate_response(indices)

    def similarity_search(self, query_embedding):
        index = faiss.IndexFlatL2(len(query_embedding[0]))
        index.add(np.array(self.vector_database).astype('float32'))
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k=5)
        return distances, indices[0]

    def generate_response(self, indices):
        responses = [self.metadata[i] for i in indices]
        return " ".join(responses)

# Example usage
rag_pipeline = RAGPipeline()
rag_pipeline.crawl_and_scrape(['https://example.com'])
response = rag_pipeline.query("What is the main topic of the website?")
print(response)
