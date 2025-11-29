import numpy as np
from .config import QUERY_CATEGORY_MAP, TOP_K_RESULTS, ROCCHIO_ALPHA, ROCCHIO_BETA, ROCCHIO_GAMMA

def is_relevant(doc_categories, query_text):
    """
    Checks if a document is relevant to a query based on its categories.
    
    Args:
        doc_categories (str): Space-separated string of categories (e.g., "hep-th quant-ph").
        query_text (str): The query string.
        
    Returns:
        bool: True if relevant, False otherwise.
    """
    target_cats = QUERY_CATEGORY_MAP.get(query_text, [])
    if not target_cats:
        return False
    
    # Split the category string into a list (e.g., "hep-th quant-ph" -> ["hep-th", "quant-ph"])
    # We use str() to handle potential NaN or non-string values safely
    doc_cat_list = str(doc_categories).split()
    
    # Check if any of the target categories are present in the document's categories
    for t_cat in target_cats:
        if t_cat in doc_cat_list:
            return True
    return False

def calculate_metrics(results, query, df, k=TOP_K_RESULTS):
    """
    Calculates Precision@k (P@k) and Average Precision (AP) for a query.
    
    Args:
        results (list): List of (doc_index, score) tuples.
        query (str): The query text.
        df (pd.DataFrame): The dataset containing categories.
        k (int): The rank at which to calculate precision.
        
    Returns:
        tuple: (p_at_k, average_precision)
    """
    relevant_count = 0
    precisions = []
    
    for rank, (doc_idx, score) in enumerate(results):
        doc_cats = df.iloc[doc_idx]['categories']
        
        if is_relevant(doc_cats, query):
            relevant_count += 1
            # Precision at this rank = (relevant found so far) / (current rank + 1)
            current_precision = relevant_count / (rank + 1)
            precisions.append(current_precision)
            
    # Precision @ k
    p_at_k = relevant_count / k
    
    # Average Precision (AP)
    if relevant_count == 0:
        ap = 0.0
    else:
        ap = sum(precisions) / relevant_count
        
    return p_at_k, ap

def rocchio_update(query_vec, X, top_indices, doc_categories, query_text):
    """
    Applies the Rocchio algorithm to refine the query vector.
    Formula: Q_new = alpha * Q_old + beta * mean(Relevant_Docs) - gamma * mean(Non_Relevant_Docs)
    
    Args:
        query_vec (sparse matrix): Original query vector.
        X (sparse matrix): Document corpus matrix.
        top_indices (list): Indices of the top retrieved documents.
        doc_categories (pd.Series): Categories column from the dataframe.
        query_text (str): The query text.
        
    Returns:
        numpy.ndarray: The new, refined query vector.
    """
    relevant_indices = []
    non_relevant_indices = []
    
    # Separate top results into relevant and non-relevant sets
    for idx in top_indices:
        cats = doc_categories.iloc[idx]
        if is_relevant(cats, query_text):
            relevant_indices.append(idx)
        else:
            non_relevant_indices.append(idx)
            
    # Calculate the new query vector
    # Start with the original query weighted by alpha
    q_new = ROCCHIO_ALPHA * query_vec
    
    # Add the centroid of relevant documents (if any)
    if relevant_indices:
        Dpos = X[relevant_indices]
        mean_pos = np.array(Dpos.mean(axis=0))
        q_new = q_new + ROCCHIO_BETA * mean_pos
        
    # Subtract the centroid of non-relevant documents (if any)
    if non_relevant_indices:
        Dneg = X[non_relevant_indices]
        mean_neg = np.array(Dneg.mean(axis=0))
        q_new = q_new - ROCCHIO_GAMMA * mean_neg
        
    # Rectify: Ensure no negative values in the vector (TF-IDF cannot be negative)
    q_new[q_new < 0] = 0
    
    return q_new
