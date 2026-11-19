import unittest
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from main import (
    clean_data,
    analyze_sentiment,
    encode_categorical,
    one_hot_encode,
    train_logistic_regression,
    predict_satisfaction
)


class TestDataProcessingPipeline(unittest.TestCase):
    """Tests data wrangling and cleaning rules."""

    def setUp(self):
        self.sample_df = pd.DataFrame([{
            'Item Purchased': '  Jeans  ',
            'Category': 'Outerwear',
            'Shipping Type': 'Express',
            'Size': 'm',
            'Review Rating': 4.5
        }])

    def test_cleaning_strips_whitespace_and_normalizes_categories(self):
        cleaned = clean_data(self.sample_df)
        self.assertEqual(cleaned['Item Purchased'].iloc[0], 'Jeans')
        self.assertEqual(cleaned['Category'].iloc[0], 'Outwear')
        self.assertEqual(cleaned['Shipping Type'].iloc[0], 'express')
        self.assertEqual(cleaned['Size'].iloc[0], 'M')
        self.assertIn('Feedback', cleaned.columns)


class TestSentimentAnalysis(unittest.TestCase):
    """Tests VADER sentiment scoring and binary satisfaction labeling."""

    def test_positive_feedback_yields_satisfaction_1(self):
        df = pd.DataFrame([{'Feedback': 'Outstanding product quality and fast delivery'}])
        res = analyze_sentiment(df)
        self.assertEqual(res['satisfaction'].iloc[0], 1)
        self.assertGreaterEqual(res['Sentiment_compound'].iloc[0], 0.0)

    def test_negative_feedback_yields_satisfaction_0(self):
        df = pd.DataFrame([{'Feedback': 'Very poor quality, terrible and disappointed'}])
        res = analyze_sentiment(df)
        self.assertEqual(res['satisfaction'].iloc[0], 0)
        self.assertLess(res['Sentiment_compound'].iloc[0], 0.0)


class TestModelTrainingAndInference(unittest.TestCase):
    """Tests model training, serialization, and live inference."""

    def setUp(self):
        self.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_customer_data.csv")

    def test_model_artifact_generation_and_inference(self):
        if not os.path.exists(self.data_path):
            self.skipTest("Sample data not present")

        df = pd.read_csv(self.data_path)
        cleaned = clean_data(df)
        sentiment_df = analyze_sentiment(cleaned)
        encoded = encode_categorical(sentiment_df)
        encoded = one_hot_encode(encoded)

        test_artifact_path = "models/test_satisfaction_model.pkl"
        model = train_logistic_regression(encoded, save_path=test_artifact_path, test_size=0.3)
        self.assertIsNotNone(model)
        self.assertTrue(os.path.exists(test_artifact_path))

        # Test inference
        sample_input = df.head(2)
        preds = predict_satisfaction(test_artifact_path, sample_input)
        self.assertEqual(len(preds), 2)
        self.assertIn(preds[0], [0, 1])

        # Cleanup test artifact
        if os.path.exists(test_artifact_path):
            os.remove(test_artifact_path)


if __name__ == "__main__":
    unittest.main()
