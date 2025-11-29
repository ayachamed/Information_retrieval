import pandas as pd
import json
import numpy as np
import os
import re
import nltk
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
JSON_FILE = 'arxiv-metadata-oai-snapshot.json'
CSV_FILE = 'arxiv_sample.csv'
INDEX_FILE = 'inverted_index.json'

# Data Loading Parameters
MAX_ROWS = 10000  # Number of rows to load from the JSON file

# Search Parameters
TOP_K_RESULTS = 10  # Number of top documents to retrieve

# Rocchio Algorithm Parameters
ROCCHIO_ALPHA = 1.0  # Weight for original query
ROCCHIO_BETA = 0.75  # Weight for relevant documents
ROCCHIO_GAMMA = 0.15 # Weight for non-relevant documents

# The 8 queries defined for the project
QUERIES = [
    "quantum field theory",
    "semiconductor laser",
    "graph neural network",
    "cosmic microwave background",
    "optical cavity",
    "spintronics",
    "superconducting qubits",
    "photonic integrated circuits"
]

# Mapping queries to their expected ArXiv categories for relevance judgment
QUERY_CATEGORY_MAP = {
    "quantum field theory": ["hep-th", "quant-ph"],
    "semiconductor laser": ["physics.optics", "cond-mat"],
    "graph neural network": ["cs.LG", "cs.AI", "stat.ML"],
    "cosmic microwave background": ["astro-ph"],
    "optical cavity": ["physics.optics", "quant-ph"],
    "spintronics": ["cond-mat"],
    "superconducting qubits": ["quant-ph"],
    "photonic integrated circuits": ["physics.optics", "eess.IV"]
}

# ==========================================
# SETUP & UTILS
# ==========================================
def download_nltk_dependencies():
    """
    Checks for and downloads necessary NLTK data packages.
    Using 'quiet=True' to avoid cluttering the console if already installed.
    """
    resources = ['stopwords', 'punkt', 'punkt_tab']
    print("Checking NLTK dependencies...")
    for res in resources:
        nltk.download(res, quiet=True)

# Initialize NLTK
download_nltk_dependencies()

# ==========================================
# PART 1: DATA LOADING
# ==========================================
def load_or_create_dataset():
    """
    Loads the dataset from a CSV file if it exists.
    Otherwise, creates a sample CSV from the large ArXiv JSON file.
    
    Returns:
        pd.DataFrame: DataFrame containing 'title', 'abstract', 'categories', and 'text'.
    """
    if os.path.exists(CSV_FILE):
        print(f"Loading existing dataset from {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
    else:
        print(f"CSV not found. Generating sample from {JSON_FILE} (Max {MAX_ROWS} rows)...")
        if not os.path.exists(JSON_FILE):
            raise FileNotFoundError(f"Please download {JSON_FILE} from Kaggle first.")
        
        rows = []
        with open(JSON_FILE, 'r') as f:
            for i, line in enumerate(f):
                if i >= MAX_ROWS: 
                    break
                d = json.loads(line)
                rows.append((d['title'], d['abstract'], d['categories']))
        
        df = pd.DataFrame(rows, columns=['title', 'abstract', 'categories'])
        df.to_csv(CSV_FILE, index=False)
        print(f"Dataset saved to {CSV_FILE}")

    # Pre-processing: Combine Title and Abstract into a single 'text' column for searching
    # We fill NaNs with empty strings to avoid errors during concatenation
    df['text'] = (df['title'].fillna('') + '. ' + df['abstract'].fillna(''))
    return df

# ==========================================
# PART 2: INVERTED INDEX (Task 1)
# ==========================================
def build_inverted_index(docs):
    """
    Builds a manual inverted index mapping terms to documents.
    Structure: {term: {doc_id: frequency}}
    
    Args:
        docs (list): List of document strings.
    """
    print("Building Inverted Index...")
    inverted_index = defaultdict(dict)
    
    stop_words = set(stopwords.words('english'))
    tokenizer = nltk.RegexpTokenizer(r'\w+')

    for doc_id, text in enumerate(docs):
        # Normalize text: lowercase and tokenize
        tokens = tokenizer.tokenize(text.lower())
        
        for token in tokens:
            # Filter out stop words and very short words
            if token not in stop_words and len(token) > 2:
                if doc_id not in inverted_index[token]:
                    inverted_index[token][doc_id] = 0
                inverted_index[token][doc_id] += 1
    
    # Save the index to a JSON file
    with open(INDEX_FILE, 'w') as f:
        json.dump(inverted_index, f)
    print(f"Inverted index saved to {INDEX_FILE} ({len(inverted_index)} terms)")

# ==========================================
# PART 3: SEARCH ENGINE (Task 2 & 3)
# ==========================================
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

# ==========================================
# PART 4: EVALUATION (Task 4)
# ==========================================
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

# ==========================================
# PART 5: ROCCHIO FEEDBACK (Task 5)
# ==========================================
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

# ==========================================
# PART 6: ABLATION STUDY (Task 6)
# ==========================================
def custom_tokenizer(text):
    """
    Custom tokenizer that stems words and removes non-alphanumeric tokens.
    """
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    # Keep only alphanumeric tokens and stem them
    stems = [stemmer.stem(t) for t in tokens if t.isalnum()]
    return stems

def run_ablation(df, queries):
    """
    Runs an ablation study to compare different text processing pipelines.
    Returns:
        list: List of tuples (pipeline_name, mean_map)
    """
    print("\n=== Running Ablation Study (Task 6) ===")
    
    # Define different configurations to test
    pipelines = [
        ("No Stop, No Stem", None, None),
        ("With Stop, No Stem", 'english', None),
        ("No Stop, With Stem", None, custom_tokenizer),
        ("With Stop, With Stem", 'english', custom_tokenizer)
    ]
    
    results_table = []

    for name, stop_w, tokenizer_func in pipelines:
        print(f"Training Pipeline: {name}...")
        
        # Initialize vectorizer with current configuration
        vec = TfidfVectorizer(stop_words=stop_w, tokenizer=tokenizer_func, min_df=3)
        
        try:
            X_ablation = vec.fit_transform(df['text'])
        except ValueError:
            print(f"Skipping {name}: Empty vocabulary.")
            continue
            
        map_scores = []
        for q in queries:
            # reuse perform_search for consistency
            results = perform_search(q, vec, X_ablation, k=TOP_K_RESULTS)
            
            # Calculate MAP
            _, ap = calculate_metrics(results, q, df, k=TOP_K_RESULTS)
            map_scores.append(ap)
            
        mean_map = np.mean(map_scores)
        results_table.append((name, mean_map))
        
    return results_table

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # 1. Load Data
    df = load_or_create_dataset()
    docs = df['text'].tolist()
    
    # 2. Build Inverted Index (Task 1)
    # Uncomment the line below if you need to regenerate the inverted index JSON
    # build_inverted_index(docs)
    
    # 3. Vectorization (Standard Pipeline)
    print("Vectorizing Documents (TF-IDF)...")
    # Standard configuration: English stopwords, min document frequency of 3
    vec = TfidfVectorizer(stop_words='english', min_df=3)
    X = vec.fit_transform(docs)
    print(f"Vocabulary size: {len(vec.get_feature_names_out())}")
    
    # 4. Run Queries & Evaluate & Rocchio (Tasks 3, 4, 5)
    print("\n=== Executing Queries (Standard vs Rocchio) ===")
    
    map_base_scores = []
    map_rocchio_scores = []
    
    query_results_data = []

    for q in QUERIES:
        # -- A. Standard Search --
        results_base = perform_search(q, vec, X, k=TOP_K_RESULTS)
        p10_base, ap_base = calculate_metrics(results_base, q, df, k=TOP_K_RESULTS)
        map_base_scores.append(ap_base)
        
        # -- B. Rocchio Feedback --
        # Get the original query vector
        q_vec = vec.transform([q])
        
        # Get indices of the top results to identify relevant/non-relevant docs
        top_indices = [idx for idx, score in results_base]
        
        # Compute new query vector using Rocchio
        q_vec_new = rocchio_update(q_vec, X, top_indices, df['categories'], q)
        
        # Search again with the new vector
        # Note: We manually compute similarity here because perform_search takes a string query
        sims_new = cosine_similarity(np.asarray(q_vec_new), X).ravel()
        top_indices_new = np.argsort(sims_new)[::-1][:TOP_K_RESULTS]
        results_rocchio = list(zip(top_indices_new, sims_new[top_indices_new]))
        
        # -- C. Evaluate Rocchio Results --
        p10_rocchio, ap_rocchio = calculate_metrics(results_rocchio, q, df, k=TOP_K_RESULTS)
        map_rocchio_scores.append(ap_rocchio)
        
        query_results_data.append({
            "Query": q,
            "P@10 (Base)": p10_base,
            "P@10 (Rocchio)": p10_rocchio
        })

    # Save Query Results to CSV
    df_query_results = pd.DataFrame(query_results_data)
    df_query_results.to_csv("query_results.csv", index=False)
    print("Saved query_results.csv")

    # Save Overall Metrics to CSV
    mean_map_base = np.mean(map_base_scores)
    mean_map_rocchio = np.mean(map_rocchio_scores)
    
    df_overall = pd.DataFrame([
        {"Metric": "Mean Average Precision (MAP) Base", "Value": mean_map_base},
        {"Metric": "Mean Average Precision (MAP) Rocchio", "Value": mean_map_rocchio}
    ])
    df_overall.to_csv("overall_metrics.csv", index=False)
    print("Saved overall_metrics.csv")
    
    # 5. Ablation Study (Task 6)
    ablation_data = run_ablation(df, QUERIES)
    
    # Save Ablation Results to CSV
    df_ablation = pd.DataFrame(ablation_data, columns=["Pipeline", "MAP"])
    df_ablation.to_csv("ablation_results.csv", index=False)
    print("Saved ablation_results.csv")

if __name__ == "__main__":
    main()