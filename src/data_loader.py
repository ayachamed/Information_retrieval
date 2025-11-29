import os
import json
import pandas as pd
import nltk
from .config import CSV_FILE, JSON_FILE, MAX_ROWS

def download_nltk_dependencies():
    """
    Checks for and downloads necessary NLTK data packages.
    Using 'quiet=True' to avoid cluttering the console if already installed.
    """
    resources = ['stopwords', 'punkt', 'punkt_tab']
    for res in resources:
        nltk.download(res, quiet=True)

def load_or_create_dataset():
    """
    Loads the dataset from a CSV file if it exists.
    Otherwise, creates a sample CSV from the large ArXiv JSON file.
    
    Returns:
        pd.DataFrame: DataFrame containing 'title', 'abstract', 'categories', and 'text'.
    """
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        if not os.path.exists(JSON_FILE):
            raise FileNotFoundError(f"Please download {JSON_FILE} from Kaggle first and place it in the data/ directory.")
        
        rows = []
        with open(JSON_FILE, 'r') as f:
            for i, line in enumerate(f):
                if i >= MAX_ROWS: 
                    break
                d = json.loads(line)
                rows.append((d['title'], d['abstract'], d['categories']))
        
        df = pd.DataFrame(rows, columns=['title', 'abstract', 'categories'])
        df.to_csv(CSV_FILE, index=False)

    # Pre-processing: Combine Title and Abstract into a single 'text' column for searching
    # We fill NaNs with empty strings to avoid errors during concatenation
    df['text'] = (df['title'].fillna('') + '. ' + df['abstract'].fillna(''))
    return df
