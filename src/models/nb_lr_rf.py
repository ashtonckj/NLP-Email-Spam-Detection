# Hybrid comparison: Naive Bayes, Logistic Regression, Random Forest, and a soft-voting ensemble
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.figures.figures_data_analysis import fig_confusion_matrix
from src.preprocessing.preprocessing import clean_text_heavy, load_split

SPAM_CSV = root_dir / "data" / "processed" / "spam.csv"
SAVE_DIR = root_dir / "src" / "saved_models"

def evaluate_model(name, y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

X_train, X_test, y_train, y_test = load_split(SPAM_CSV)

X_train_text = X_train["Message"].apply(lambda t: clean_text_heavy(t, ps, stop_words))
X_test_text = X_test["Message"].apply(lambda t: clean_text_heavy(t, ps, stop_words))

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

results = []

# --- Naive Bayes ---
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
results.append(evaluate_model("Naive Bayes", y_test, nb.predict(X_test_vec)))

# --- Logistic Regression ---
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_vec, y_train)
results.append(evaluate_model("Logistic Regression", y_test, lr.predict(X_test_vec)))

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_vec, y_train)
results.append(evaluate_model("Random Forest", y_test, rf.predict(X_test_vec)))

# --- Hybrid: NB + LR + RF soft-voting ensemble ---
hybrid = VotingClassifier(estimators=[("nb", nb), ("lr", lr), ("rf", rf)], voting="soft")
hybrid.fit(X_train_vec, y_train)
y_pred = hybrid.predict(X_test_vec)
results.append(evaluate_model("NB + LR + RF Hybrid", y_test, y_pred))
fig_confusion_matrix(y_test, y_pred, "nb_lr_rf")

summary = pd.DataFrame(results).set_index("model")
print("\n=== Summary ===")
print(summary.round(4))
SAVE_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(hybrid, SAVE_DIR / "nb_lr_rf__hybrid.joblib")
joblib.dump(vectorizer, SAVE_DIR / "nb_lr_rf__tfidf_vectorizer.joblib")

metrics_out = {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in results}
with open(SAVE_DIR / "nb_lr_rf_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"\nSaved hybrid model, vectorizer, and metrics to {SAVE_DIR}")