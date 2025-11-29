import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .config import TOP_K_RESULTS

def perform_search(query, vectorizer, X, k=TOP_K_RESULTS):
    """
    Performs a search using Cosine Similarity on TF-IDF vectors.
    
    Args:
        query (str): The user's search query.
        vectorizer (TfidfVectorizer): Fitted vectorizer.
        X (sparse matrix): TF-IDF matrix of the corpus.
        k (int): Number of results to return.
        
    Returns:
        list: List of tuples (document_index, similarity_score).
    """
    # Convert query to the same TF-IDF vector space as the documents
    query_vec = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and all documents
    # flatten() or ravel() converts the result to a 1D array
    similarities = cosine_similarity(query_vec, X).ravel()
    
    # Get indices of the top k documents (sorted by score descending)
    top_indices = np.argsort(similarities)[::-1][:k]
    
    # Return pairs of (index, score)
    return list(zip(top_indices, similarities[top_indices]))
