

# Faisla.ai - AI-Assisted Legal Intelligence System

### *NLP – Semester Project*

**Members:** Sajeel Tariq, Faisal Shahid, Hunaiza Khan, Syed Anas Ahmed

---

## 📌 Overview

This project builds an AI-powered system to support legal workflows in Pakistan by addressing delays, backlogs, and inefficient case management. The system consists of three integrated components: **Case Classification AI**, a **Legal Precedent Search Tool**, and a **Case Prioritization Model**. These modules use NLP, machine learning, and hybrid retrieval techniques to analyze legal documents, classify cases, retrieve precedents, and estimate case influence.

---

## 📂 Dataset

The project uses the **Legal Case Reports in Australia (2006–2009)** dataset from the UCI ML Repository (re-hosted on Kaggle). It contains XML case files from the Federal Court of Australia, including catchphrases, sentences, annotations, and structured metadata. The dataset is widely used in research for legal summarization, citation analysis, and document classification.

Kaggle link: https://www.kaggle.com/datasets/thedevastator/legal-case-reports-in-australia-2006-2009

---

# 🧩 Sub-Problem 1: Case Classification AI

This module classifies legal cases into meaningful categories using a combination of topic modeling and supervised learning.

### **Methodology**

* **XML Parsing:** Extracted case name, catchphrases, and full text using BeautifulSoup and stored in a DataFrame.
* **Topic Modeling:** Used BERTopic with `all-MiniLM-L6-v2` embeddings to identify latent topics from catchphrases. Topics were assigned to cases by averaging topic probabilities; unclear cases labeled *Miscellaneous*.
* **Category Mapping:** Topics manually mapped to domains like Immigration, Taxation, Employment, IP, Corporate, Environment, Compensation, etc.
* **Preprocessing:** Combined catchphrases + text into `full_text`. Missing values replaced with empty strings.
* **TF-IDF Vectorization:**

  * `stop_words="english"`
  * `max_features=10000`
  * `ngram_range=(1,2)`
* **Training & Evaluation:**

  * Train/test split (80/20) with stratification
  * Logistic Regression (`class_weight="balanced"`, `max_iter=200`)
  * Evaluated via classification report and confusion matrix heatmap

### **Outcome**

Produced a structured classification dataset and a robust multi-class classifier enabling case categorization for downstream legal AI modules.

---

# 🔎 Sub-Problem 2: Hybrid Legal Precedent Search Tool

This module retrieves relevant legal text segments using combined semantic, lexical, and reranking methods.

### **Methodology**

* **Embedding Model:** Generated 384-dim vectors using `all-MiniLM-L6-v2`.
* **Chunking:** Split documents into ~250-word chunks for efficient semantic retrieval.
* **Data Cleaning:** Fixed malformed XML tags, normalized encoding to UTF-8.
* **FAISS Index:** Created a FlatL2 vector index storing embeddings with mappings to filenames and text chunks.
* **Whoosh + BM25:** Built a lexical search index for keyword-based retrieval; auto-reloads if already built.
* **Hybrid Search:**

  1. Retrieve results via FAISS (semantic)
  2. Retrieve via BM25 (lexical)
  3. Merge results with weighted scoring
* **Reranking:** Used `ms-marco-MiniLM-L-6-v2` cross-encoder for contextual relevance.
* **Interactive Query Loop:** Accepts natural language queries and displays ranked results with file name, relevance score, and text snippet.

### **Outcome**

A scalable, high-precision legal retrieval system combining meaning-based and keyword-based search, suitable for precedent discovery in large corpora.

---

# 📊 Sub-Problem 3: Case Prioritization Model

This module estimates the “influence score” of legal cases using heuristics and machine learning.

### **Methodology**

* **Parsing:** Extracted case name, catchphrases, and sentences into CSV.
* **Feature Engineering:** Calculated sentence count, word count, keyword frequencies, court level, year, and recency (current year − case year).
* **Heuristic Scoring:** Designed a formula (inspired by Journal of Harbin Engineering University) combining hierarchy, text richness, keyword density, and case age. Generated `heuristic_ranking.csv`.
* **BERT Embeddings:** Used `bert-base-uncased` to embed case names and catchphrases.
* **Model Training:**

  * Random Forest Regressor
  * Inputs: numerical + semantic features
  * Output: predicted influence score
* **Evaluation:** Used R² and MAE; compared predicted scores with heuristic scores.

### **Outcome**

A refined, data-driven ranking of case influence, useful for applications like citation prediction, legal trend analysis, and decision support systems.

---

## 📁 Key Outputs

* `parsed_cases.csv`
* `case_classification_dataset.csv`
* `heuristic_ranking.csv`
* FAISS index + mapping files
* Whoosh BM25 index
* `case_influence_rf.joblib`
* Saved BERT models and tokenizers

---

## 🛠️ Tech Stack

Python, BeautifulSoup, SentenceTransformers, BERTopic, Scikit-learn, FAISS, Whoosh, HuggingFace Transformers, RandomForestRegressor.

---

## 🛠️ Video

https://www.youtube.com/watch?v=mYTOdPlc9kc&t=1s
---
