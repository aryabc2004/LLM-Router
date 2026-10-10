"""
Trains the complexity classifier and saves it to classifier.joblib.

Run this once before starting the API:
    python train_classifier.py
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

from complexity_classifier import ComplexityClassifier, hand_features

DATA_PATH = Path(__file__).parent / "data" / "training_prompts.csv"


def main():
    if not DATA_PATH.exists():
        print("No training data found, generating it...")
        subprocess.run([sys.executable, str(Path(__file__).parent / "data" / "build_dataset.py")], check=True)

    df = pd.read_csv(DATA_PATH)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["prompt"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    tfidf_train = vectorizer.fit_transform(X_train_text)
    tfidf_test = vectorizer.transform(X_test_text)

    scaler = StandardScaler()
    hand_train = scaler.fit_transform(hand_features(X_train_text.tolist()))
    hand_test = scaler.transform(hand_features(X_test_text.tolist()))

    X_train = hstack([tfidf_train, csr_matrix(hand_train)])
    X_test = hstack([tfidf_test, csr_matrix(hand_test)])

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    clf = ComplexityClassifier(vectorizer=vectorizer, scaler=scaler, model=model)
    clf.save()
    print(f"Saved trained classifier to {clf.MODEL_PATH if hasattr(clf, 'MODEL_PATH') else 'classifier.joblib'}")


if __name__ == "__main__":
    main()
