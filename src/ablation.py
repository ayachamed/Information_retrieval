import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from .search import perform_search
from .evaluation import calculate_metrics
from .config import TOP_K_RESULTS

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
    # Define different configurations to test
    pipelines = [
        ("No Stop, No Stem", None, None),
        ("With Stop, No Stem", 'english', None),
        ("No Stop, With Stem", None, custom_tokenizer),
        ("With Stop, With Stem", 'english', custom_tokenizer)
    ]
    
    results_table = []

    for name, stop_w, tokenizer_func in pipelines:
        # Initialize vectorizer with current configuration
        vec = TfidfVectorizer(stop_words=stop_w, tokenizer=tokenizer_func, min_df=3)
        
        try:
            X_ablation = vec.fit_transform(df['text'])
        except ValueError:
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
