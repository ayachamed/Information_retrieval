import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    QUERIES, TOP_K_RESULTS, 
    QUERY_RESULTS_FILE, OVERALL_METRICS_FILE, ABLATION_RESULTS_FILE
)
from src.data_loader import load_or_create_dataset, download_nltk_dependencies
from src.index import build_inverted_index
from src.search import perform_search
from src.evaluation import calculate_metrics, rocchio_update
from src.ablation import run_ablation

def main():
    # Initialize NLTK
    download_nltk_dependencies()

    # 1. Load Data
    df = load_or_create_dataset()
    docs = df['text'].tolist()
    
    # 2. Build Inverted Index (Task 1)
    # build_inverted_index(docs)
    
    # 3. Vectorization (Standard Pipeline)
    # Standard configuration: English stopwords, min document frequency of 3
    vec = TfidfVectorizer(stop_words='english', min_df=3)
    X = vec.fit_transform(docs)
    
    # 4. Run Queries & Evaluate & Rocchio (Tasks 3, 4, 5)
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
    df_query_results.to_csv(QUERY_RESULTS_FILE, index=False)

    # Save Overall Metrics to CSV
    mean_map_base = np.mean(map_base_scores)
    mean_map_rocchio = np.mean(map_rocchio_scores)
    
    df_overall = pd.DataFrame([
        {"Metric": "Mean Average Precision (MAP) Base", "Value": mean_map_base},
        {"Metric": "Mean Average Precision (MAP) Rocchio", "Value": mean_map_rocchio}
    ])
    df_overall.to_csv(OVERALL_METRICS_FILE, index=False)
    
    # 5. Ablation Study (Task 6)
    ablation_data = run_ablation(df, QUERIES)
    
    # Save Ablation Results to CSV
    df_ablation = pd.DataFrame(ablation_data, columns=["Pipeline", "MAP"])
    df_ablation.to_csv(ABLATION_RESULTS_FILE, index=False)

if __name__ == "__main__":
    main()