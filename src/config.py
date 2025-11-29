import os

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

JSON_FILE = os.path.join(DATA_DIR, 'arxiv-metadata-oai-snapshot.json')
CSV_FILE = os.path.join(DATA_DIR, 'arxiv_sample.csv')
INDEX_FILE = os.path.join(DATA_DIR, 'inverted_index.json')

QUERY_RESULTS_FILE = os.path.join(RESULTS_DIR, 'query_results.csv')
OVERALL_METRICS_FILE = os.path.join(RESULTS_DIR, 'overall_metrics.csv')
ABLATION_RESULTS_FILE = os.path.join(RESULTS_DIR, 'ablation_results.csv')

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
