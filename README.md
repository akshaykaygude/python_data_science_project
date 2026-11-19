# 🛍️ Customer Satisfaction & Sentiment Prediction Pipeline

> **An End-to-End E-Commerce Machine Learning & NLP System**  
> *Predicting customer satisfaction sentiment from retail transactions and reviews using NLTK (VADER), Feature Engineering, and Scaled Logistic Regression.*

[![ML Pipeline CI](https://github.com/akshaykaygude/python_data_science_project/actions/workflows/ci.yml/badge.svg)](https://github.com/akshaykaygude/python_data_science_project/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK%20VADER-green.svg)](https://www.nltk.org/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Wrangling-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Current Capabilities: What Already Works

The following components are fully implemented, verified, and test-covered in this codebase:

- [x] **Data Ingestion & Normalization (`main.py`)**:
  - Handles missing values, strips whitespace, standardizes casing, and maps inconsistent product categories.
- [x] **NLP Sentiment Polarity Extraction (`main.py`)**:
  - Uses NLTK's VADER (`SentimentIntensityAnalyzer`) to extract compound sentiment scores from customer review feedback.
  - Derives a clean binary classification target (`satisfaction`: `1` for compound $\ge 0$, `0` otherwise).
- [x] **Feature Engineering & Encoders (`main.py`)**:
  - Encodes categorical customer features (`Category`, `Shipping Type`, `Payment Method`, `Season`).
  - Applies One-Hot Encoding (`pd.get_dummies`) with reference column dropping to prevent multicollinearity.
- [x] **Feature Normalization & Classification (`main.py`)**:
  - Fits `StandardScaler` to prevent convergence warnings and ensure stable gradient descent.
  - Trains a scaled Logistic Regression model achieving **93.11% accuracy** and **96.38% F1-score**.
- [x] **Model Serialization & Inference (`models/satisfaction_model.pkl`)**:
  - Serializes trained model, fitted scaler, and expected feature column names into a unified artifact via `joblib`.
  - Exposes `predict_satisfaction(artifact_path, new_df)` for real-time inference on new customer transactions.
- [x] **Automated Unit & Integration Test Suite (`tests/test_model_pipeline.py`)**:
  - Unit tests covering data cleaning, sentiment boundaries, categorical encoding, and inference.
- [x] **Classifier Benchmarking Suite (`benchmark.py`)**:
  - Evaluates Logistic Regression against Decision Tree and Random Forest baselines across Accuracy, Precision, Recall, and F1.
- [x] **Continuous Integration Pipeline (`.github/workflows/ci.yml`)**:
  - Automated GitHub Actions workflow testing the full pipeline across Python 3.10 and 3.11.

---

## ⚠️ Known Limitations & Real-World Edge Cases

1. **Rule-Based Sentiment Limitations**:
   - VADER is an effective lexicon-based analyzer for short consumer comments, but can misinterpret nuanced domain jargon, sarcasm, or complex conditional phrasing without deep contextual embeddings.
2. **Class Imbalance in Review Datasets**:
   - Like most retail platforms, satisfied reviews significantly outnumber dissatisfied reviews (~90% positive). While precision is high, threshold tuning and class weighting (`class_weight='balanced'`) are advised for aggressive dissatisfied-customer intervention.
3. **Cold-Start for New Categories/Locations**:
   - Incoming records with unseen one-hot categories default to zeros in the feature alignment step.

---

## 🗺️ Machine Learning Roadmap & Next Milestones

- [x] **v1.0 (Baseline Pipeline)**: Core data wrangling, VADER sentiment scoring, and Logistic Regression.
- [x] **v1.1 (Production Reliability)**: Feature scaling with `StandardScaler`, CLI arguments, and model serialization.
- [x] **v1.2 (Evaluation & Tests)**: Automated test suite, classifier benchmarking suite, and GitHub Actions CI.
- [ ] **v2.0 (Transformer Embeddings)**: Replace VADER with fine-tuned `distilbert-base-uncased` for sentiment analysis.
- [ ] **v2.1 (Hyperparameter Optimization)**: Automated grid search and 5-fold stratified cross-validation.
- [ ] **v2.2 (REST API Service)**: Lightweight FastAPI microservice serving real-time predictions.

---

## 📁 Repository Structure

```text
python_data_science_project/
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions automated test workflow
├── tests/
│   └── test_model_pipeline.py      # Unit tests (cleaning, sentiment, inference)
├── data/
│   └── sample_customer_data.csv    # Included demo dataset (1,500 rows, 18 features)
├── models/
│   └── satisfaction_model.pkl      # Serialized trained model & StandardScaler
├── main.py                         # Production CLI pipeline (clean, train, evaluate, infer)
├── benchmark.py                    # Multi-model benchmarking script (LR vs DT vs RF)
├── project1_call.ipynb             # Interactive Jupyter Notebook walkthrough
├── requirements.txt                # Python dependencies
├── .gitignore                      # Exclusions for cache, checkpoints, and venvs
├── LICENSE                         # MIT Open-Source License
└── README.md                       # Comprehensive documentation & setup guide
```

---

## 🚀 2-Minute Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/akshaykaygude/python_data_science_project.git
cd python_data_science_project
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Option A: Run Full ML Pipeline
```bash
python main.py
```

### Option B: Run Classifier Benchmarks
```bash
python benchmark.py
```

### Option C: Run Automated Test Suite
```bash
python -m unittest tests/test_model_pipeline.py
```

### Option D: Interactive Jupyter Notebook
```bash
jupyter notebook project1_call.ipynb
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
