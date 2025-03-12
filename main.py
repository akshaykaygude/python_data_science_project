import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean and preprocess the dataset."""
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Drop unnecessary column safely

    # Strip and format string columns
    for col in ['Shipping Type', 'Payment Method', 'Category', 'Item Purchased',
                'Location', 'Color', 'Season', 'Size']:
        df[col] = df[col].str.strip()

    df['Shipping Type'] = df['Shipping Type'].str.lower()
    df['Size'] = df['Size'].str.upper()

    # Fix category inconsistency
    df['Category'] = df['Category'].replace('Outerwear', 'Outwear')

    # Generate feedback based on Review Rating
    df['Feedback'] = df['Review Rating'].apply(
        lambda x: np.random.choice(
            ['This is Good', 'Very good for use', 'Outstanding product quality']
        ) if x > 2.5 else np.random.choice(
            ['Bad product', 'Very poor quality', 'Disappointed with performance']
        )
    )

    return df


def analyze_sentiment(df):
    """Compute sentiment scores and satisfaction labels."""
    analyzer = SentimentIntensityAnalyzer()
    df["Sentiment_compound"] = df["Feedback"].apply(lambda text: analyzer.polarity_scores(text)["compound"])
    df["satisfaction"] = (df["Sentiment_compound"] >= 0).astype(int)
    return df


def encode_categorical(df):
    """Convert categorical columns to numerical codes."""
    cat_cols = ['Gender', 'Category', 'Size', 'Season', 'Subscription Status', 'Shipping Type',
                'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases', 'Feedback']

    df[cat_cols] = df[cat_cols].astype('category').apply(lambda x: x.cat.codes)
    return df


def one_hot_encode(df):
    """One-hot encodes predefined categorical columns and drops the first category as a reference."""
    categorical_columns = ['Item Purchased', 'Location', 'Color']  # Defined inside the function
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df


def train_logistic_regression(df):
    """Prepare data, train logistic regression, and evaluate model."""

    # Drop unnecessary columns and fill missing values
    df_filt = df.drop(columns=['Item Purchased', 'Location', 'Color', 'Sentiment_compound', 'Review Rating'],
                      errors='ignore')
    df_filt = df_filt.fillna(0)

    # Define features and target
    X = df_filt.drop(columns=['satisfaction'])
    y = df_filt['satisfaction']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Print evaluation metrics
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))

    return model  # Return trained model for further use if needed

