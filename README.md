# Scientific Abstracts Retrieval System (arXiv)

## 📌 Project Overview
This project is an **Information Retrieval (IR) System** built to index, search, and rank scientific abstracts from the **arXiv dataset**. The system utilizes the **Vector Space Model (VSM)** with **TF-IDF weighting** and implements **Rocchio Relevance Feedback** to improve search results.

It was developed as part of the **"Techniques d'Indexation et de Référencement"** course at **ISAMM**.

---

## 🚀 Features
- **Inverted Indexing:** Efficient manual construction of an inverted index mapping terms to documents.
- **Vector Space Model:** Uses TF-IDF (Term Frequency-Inverse Document Frequency) for document representation.
- **Cosine Similarity Search:** Ranks documents based on their similarity to the query vector.

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
4. **Search & Evaluation:** It runs 8 predefined queries and saves the P@10 scores to **`query_results.csv`**.
5. **Rocchio Feedback:** It refines the queries, re-runs the search, and saves the MAP scores to **`overall_metrics.csv`**.
6. **Ablation Study:** It tests 4 different preprocessing configurations and saves the results to **`ablation_results.csv`**.

---

## 📊 Experimental Results

### Performance (Base vs. Rocchio)
| Query                            | P@10 (Base) | P@10 (Rocchio) |
| -------------------------------- | ----------- | -------------- |
| Quantum field theory             | 0.70        | **0.70**       |
| Graph neural network             | 0.10        | **0.20**       |
| Superconducting qubits           | 0.50        | **0.60**       |
| **Mean Average Precision (MAP)** | **0.4919**  | **0.7766**     |

**Observation:** The Rocchio algorithm significantly improved retrieval performance by expanding queries with relevant terms found in the initial top results.

### Ablation Study
| Pipeline Configuration        | MAP Score  |
| ----------------------------- | ---------- |
| **No Stopwords, No Stemming** | **0.5350** |
| With Stopwords, No Stemming   | 0.4919     |
| No Stopwords, With Stemming   | 0.4579     |
| With Stopwords, With Stemming | 0.4506     |

**Insight:** For this specific scientific corpus, raw terms (without stemming) provided better precision than stemmed terms, likely because scientific terminology requires exact matches.

## 👥 Authors
**Mohamed Ayacha & Ahmed Kchouk**  
L3 IMM - ISAMM  
*Techniques of Information retrieval and indexation*  
**2025-2026**