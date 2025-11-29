import json
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from .config import INDEX_FILE

def build_inverted_index(docs):
    """
    Builds a manual inverted index mapping terms to documents.
    Structure: {term: {doc_id: frequency}}
    
    Args:
        docs (list): List of document strings.
    """
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
