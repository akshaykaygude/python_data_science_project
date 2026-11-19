"""
benchmark.py
Compares multiple classifier baselines for customer satisfaction prediction.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from main import load_data, clean_data, analyze_sentiment, encode_categorical, one_hot_encode


def run_benchmark(data_path="data/sample_customer_data.csv"):
    print("=" * 65)
    print("      CLASSIFICATION BENCHMARK: CUSTOMER SATISFACTION")
    print("=" * 65)

    df = load_data(data_path)
    df = clean_data(df)
    df = analyze_sentiment(df)
    df = encode_categorical(df)
    df = one_hot_encode(df)

    df_filt = df.drop(
        columns=['Customer ID', 'Item Purchased', 'Location', 'Color', 'Sentiment_compound', 'Review Rating'],
        errors='ignore'
    ).fillna(0)

    X = df_filt.drop(columns=['satisfaction'])
    y = df_filt['satisfaction']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifiers = {
        "Logistic Regression (Current)": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest (100 Trees)": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    }

    results = []
    for name, clf in classifiers.items():
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)

        results.append({
            "Model": name,
            "Accuracy": f"{accuracy_score(y_test, preds) * 100:.2f}%",
            "Precision": f"{precision_score(y_test, preds, zero_division=0) * 100:.2f}%",
            "Recall": f"{recall_score(y_test, preds, zero_division=0) * 100:.2f}%",
            "F1-Score": f"{f1_score(y_test, preds, zero_division=0) * 100:.2f}%"
        })

    report_df = pd.DataFrame(results)
    print("\n" + report_df.to_string(index=False))
    print("=" * 65)
    return report_df


if __name__ == "__main__":
    run_benchmark()
