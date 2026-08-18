# Hybrid model: NB-SVM -- Naive Bayes log-count ratios reweight the TF-IDF
# features, then a linear SVM is trained on the reweighted features.
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

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


def nb_log_count_ratio(X_train_vec, y_train, alpha=1.0):
    """Naive Bayes log-count ratio: r = log((p / ||p||_1) / (q / ||q||_1))

    p = smoothed sum of (TF-IDF) feature weights over spam documents
    q = smoothed sum of (TF-IDF) feature weights over ham documents

    r amplifies terms whose weight is disproportionately concentrated in
    one class, which is exactly what the SVM should be paying more
    attention to.
    """
    X_train_vec = sp.csr_matrix(X_train_vec)
    y_train = np.asarray(y_train)

    p = alpha + X_train_vec[y_train == 1].sum(axis=0)
    q = alpha + X_train_vec[y_train == 0].sum(axis=0)

    p = np.asarray(p).ravel()
    q = np.asarray(q).ravel()

    r = np.log((p / p.sum()) / (q / q.sum()))
    return r


ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Same train/test split every model uses
X_train, X_test, y_train, y_test = load_split(SPAM_CSV)

# Heavy clean (stopwords removed, stemmed) -- tfidf branch only
X_train_text = X_train["Message"].apply(lambda t: clean_text_heavy(t, ps, stop_words))
X_test_text = X_test["Message"].apply(lambda t: clean_text_heavy(t, ps, stop_words))

# Fit tfidf on train only, reuse the same vocab on test
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# --- Naive Bayes log-count ratio, fit on train only ---
r = nb_log_count_ratio(X_train_vec, y_train)
r_sparse = sp.diags(r)

# Reweight every TF-IDF feature column by its NB log-count ratio
X_train_nb = sp.csr_matrix(X_train_vec.dot(r_sparse))
X_test_nb = sp.csr_matrix(X_test_vec.dot(r_sparse))

# --- Linear SVM trained on the NB-reweighted features ---
svm = LinearSVC()
svm.fit(X_train_nb, y_train)
y_pred = svm.predict(X_test_nb)
results = [evaluate_model("NB-SVM", y_test, y_pred)]
fig_confusion_matrix(y_test, y_pred, "nb_svm")

summary = pd.DataFrame(results).set_index("model")
print("\n=== Summary ===")
print(summary.round(4))
SAVE_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"vectorizer": vectorizer, "nb_log_count_ratio": r, "svm": svm},
    SAVE_DIR / "nb_svm.joblib",
)

metrics_out = {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in results}
with open(SAVE_DIR / "nb_svm_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"\nSaved NB-SVM bundle and metrics to {SAVE_DIR}")