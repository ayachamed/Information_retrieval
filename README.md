# Scientific Abstracts Retrieval System (arXiv)

## 📌 Project Overview
This project is an **Information Retrieval (IR) System** built to index, search, and rank scientific abstracts from the **arXiv dataset**. The system utilizes the **Vector Space Model (VSM)** with **TF-IDF weighting** and implements **Rocchio Relevance Feedback** to improve search results.

It was developed as part of the **"Techniques d'Indexation et de Référencement"** course at **ISAMM**.

---

## 🚀 Features
- **Inverted Indexing:** Efficient manual construction of an inverted index mapping terms to documents.
- **Vector Space Model:** Uses TF-IDF (Term Frequency-Inverse Document Frequency) for document representation.
- **Cosine Similarity Search:** Ranks documents based on their similarity to the query vector.
- **Rocchio Algorithm:** Implements query expansion/refinement based on pseudo-relevance feedback.
- **Evaluation Metrics:** Automatically calculates **Precision@K (P@K)** and **Mean Average Precision (MAP)**.
- **Ablation Study:** Compares different text preprocessing pipelines (Stemming vs. No Stemming, Stopwords vs. No Stopwords).

---

## 📂 Project Structure
```
├── arxiv-metadata-oai-snapshot.json  # Raw Dataset (Download from Kaggle)
├── arxiv_sample.csv                  # Processed Sample Dataset (Generated)
├── inverted_index.json               # Generated Inverted Index
├── main.py                           # Main Source Code
├── README.md                         # Project Documentation
└── report.pdf                        # Final Project Report
```

---

## 🛠️ Prerequisites & Installation

### 1. Requirements
Ensure you have **Python 3.8+** installed. You will need the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`

### 2. Installation
Install dependencies via pip:
```bash
pip install pandas numpy scikit-learn nltk
```

### 3. Dataset
The project requires the **arXiv Metadata Dataset**.
1. Download `arxiv-metadata-oai-snapshot.json` from [Kaggle](https://www.kaggle.com/Cornell-University/arxiv).
2. Place it in the project root directory.

---

## 🏃‍♂️ How to Run

Execute the main script from your terminal:

```bash
python main.py
```

### What happens when you run it?
1. **Data Loading:** It reads the JSON file and creates a CSV sample (`arxiv_sample.csv`) of 10,000 documents.
2. **Indexing:** It builds a manual inverted index and saves it to `inverted_index.json`.
3. **Vectorization:** It converts the text data into TF-IDF matrices.
4. **Search & Evaluation:** It runs 8 predefined queries (e.g., "Quantum field theory") and calculates P@10.
5. **Rocchio Feedback:** It refines the queries and re-runs the search to show performance improvements.
6. **Ablation Study:** It tests 4 different preprocessing configurations and outputs the MAP scores.

---

## 📊 Experimental Results

### Performance (Base vs. Rocchio)
| Query | P@10 (Base) | P@10 (Rocchio) |
|-------|------------|----------------|
| Quantum field theory | 0.70 | **0.70** |
| Graph neural network | 0.10 | **0.20** |
| Superconducting qubits | 0.50 | **0.60** |
| **Mean Average Precision (MAP)** | **0.4919** | **0.7766** |

**Observation:** The Rocchio algorithm significantly improved retrieval performance by expanding queries with relevant terms found in the initial top results.

### Ablation Study
| Pipeline Configuration | MAP Score |
|------------------------|-----------|
| **No Stopwords, No Stemming** | **0.5350** |
| With Stopwords, No Stemming | 0.4919 |
| No Stopwords, With Stemming | 0.4579 |
| With Stopwords, With Stemming | 0.4506 |

**Insight:** For this specific scientific corpus, raw terms (without stemming) provided better precision than stemmed terms, likely because scientific terminology requires exact matches.

## 👥 Authors
**Mohamed Ayacha & Ahmed Kchouk**  
L3 IMM - ISAMM  
*Techniques d'Indexation et de Référencement*  
**2025-2026**
```
