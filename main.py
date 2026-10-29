"""
main.py
End-to-End Customer Satisfaction & Sentiment Prediction Pipeline
----------------------------------------------------------------
1. Data Ingestion & Preprocessing
2. NLP Sentiment Analysis via NLTK VADER
3. Feature Engineering & Categorical Encoding
4. Logistic Regression Classifier Training & Evaluation
5. Model Persistence & Inference Functionality
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Download VADER lexicon quietly on first run
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[+] Loaded dataset: {file_path} ({df.shape[0]:,} rows, {df.shape[1]} columns)")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the customer transaction dataset."""
    df = df.copy()
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    # Strip and format string columns if present
    string_cols = [
        'Shipping Type', 'Payment Method', 'Category', 'Item Purchased',
        'Location', 'Color', 'Season', 'Size'
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if 'Shipping Type' in df.columns:
        df['Shipping Type'] = df['Shipping Type'].str.lower()
    if 'Size' in df.columns:
        df['Size'] = df['Size'].str.upper()

    # Fix category inconsistency
    if 'Category' in df.columns:
        df['Category'] = df['Category'].replace('Outerwear', 'Outwear')

    # Generate or standardize Feedback text based on Review Rating if not already present
    if 'Feedback' not in df.columns and 'Review Rating' in df.columns:
        df['Feedback'] = df['Review Rating'].apply(
            lambda x: np.random.choice(
                ['This is Good', 'Very good for use', 'Outstanding product quality']
            ) if float(x) > 2.5 else np.random.choice(
                ['Bad product', 'Very poor quality', 'Disappointed with performance']
            )
        )

    print("[+] Data cleaned and formatted.")
    return df


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VADER sentiment polarity scores and derive satisfaction labels."""
    df = df.copy()
    analyzer = SentimentIntensityAnalyzer()

    # If Feedback column exists, compute compound sentiment score
    if 'Feedback' in df.columns:
        df["Sentiment_compound"] = df["Feedback"].apply(
            lambda text: analyzer.polarity_scores(str(text))["compound"]
        )
        df["satisfaction"] = (df["Sentiment_compound"] >= 0).astype(int)
    elif 'Review Rating' in df.columns:
        # Fallback directly to review rating threshold
        df["satisfaction"] = (df["Review Rating"] >= 3.0).astype(int)

    satisfied_count = (df["satisfaction"] == 1).sum()
    print(f"[+] Sentiment analysis complete: {satisfied_count:,} satisfied (1), "
          f"{len(df) - satisfied_count:,} dissatisfied (0).")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to numerical category codes."""
    df = df.copy()
    cat_cols = [
        'Gender', 'Category', 'Size', 'Season', 'Subscription Status', 'Shipping Type',
        'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases', 'Feedback'
    ]
    present_cols = [c for c in cat_cols if c in df.columns]
    df[present_cols] = df[present_cols].astype('category').apply(lambda x: x.cat.codes)
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes nominal columns and drops the first category as reference."""
    df = df.copy()
    categorical_columns = ['Item Purchased', 'Location', 'Color']
    present_cols = [c for c in categorical_columns if c in df.columns]
    df = pd.get_dummies(df, columns=present_cols, drop_first=True)
    return df


def train_logistic_regression(
    df: pd.DataFrame, 
    save_path: str = "models/satisfaction_model.pkl",
    test_size: float = 0.30,
    random_state: int = 42
):
    """
    Prepares features, scales numeric inputs, trains Logistic Regression,
    evaluates performance, and serializes the model artifact.
    """
    df_filt = df.drop(
        columns=['Customer ID', 'Item Purchased', 'Location', 'Color', 'Sentiment_compound', 'Review Rating'],
        errors='ignore'
    )
    df_filt = df_filt.fillna(0)

    if 'satisfaction' not in df_filt.columns:
        raise ValueError("Target column 'satisfaction' not found in dataframe.")

    X = df_filt.drop(columns=['satisfaction'])
    y = df_filt['satisfaction']
    feature_names = list(X.columns)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature scaling with StandardScaler (prevents lbfgs convergence warning)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model with increased iterations
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # Predictions & Evaluation
    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    rec = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 50)
    print("           MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, predictions, zero_division=0))
    print("=" * 50)

    # Save model and artifacts
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        artifact = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names
        }
        joblib.dump(artifact, save_path)
        print(f"\n[+] Trained model & scaler successfully saved to: '{save_path}'")

    return model


def predict_satisfaction(artifact_path: str, new_df: pd.DataFrame) -> np.ndarray:
    """Inference function to predict satisfaction on new raw customer records."""
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Model artifact not found at: {artifact_path}")
    
    artifact = joblib.load(artifact_path)
    model = artifact['model']
    scaler = artifact['scaler']
    feature_names = artifact['feature_names']

    # Preprocess
    cleaned = clean_data(new_df)
    encoded = encode_categorical(cleaned)
    encoded = one_hot_encode(encoded)

    # Align columns
    for col in feature_names:
        if col not in encoded.columns:
            encoded[col] = 0
    X_input = encoded[feature_names]

    scaled = scaler.transform(X_input)
    predictions = model.predict(scaled)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Customer Satisfaction & Sentiment Prediction Pipeline")
    default_csv = os.path.join(os.path.dirname(__file__), "data", "sample_customer_data.csv")
    parser.add_argument("--data", type=str, default=default_csv, help="Path to input CSV dataset")
    parser.add_argument("--save-model", type=str, default="models/satisfaction_model.pkl", help="Path to save model artifact")
    parser.add_argument("--test-size", type=float, default=0.30, help="Test set split ratio (default: 0.30)")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("   Customer Satisfaction & Sentiment Prediction Pipeline   ")
    print("=" * 60)

    # Step 1: Load Data
    df = load_data(args.data)

    # Step 2: Clean Data
    df = clean_data(df)

    # Step 3: Sentiment Analysis
    df = analyze_sentiment(df)

    # Step 4: Feature Encoding
    df = encode_categorical(df)
    df = one_hot_encode(df)

    # Step 5: Model Training & Evaluation
    train_logistic_regression(df, save_path=args.save_model, test_size=args.test_size)


if __name__ == "__main__":
    main()
