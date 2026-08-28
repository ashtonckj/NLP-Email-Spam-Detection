# Hybrid Model: Naive Bayes log-count ratios and linear SVM
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.figures.figures_data_analysis import fig_confusion_matrix
from src.models.evaluation import evaluate_model, export_model_metrics
from src.preprocessing.preprocessing import clean_text_heavy, load_split

SPAM_CSV = ROOT_DIR / "data" / "processed" / "spam.csv"
SAVE_DIR = ROOT_DIR / "src" / "saved_models"


def nb_log_count_ratio(X_train_vec, y_train, alpha=1.0):
    """Naive Bayes log-count ratio: r = log((p / ||p||_1) / (q / ||q||_1))

    p = smoothed sum of (TF-IDF) feature weights over spam documents
    q = smoothed sum of (TF-IDF) feature weights over ham documents

    r amplifies terms whose weight is disproportionately concentrated in
    one class, which is exactly what the SVM should be paying more attention to.
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
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# --- Naive Bayes log-count ratio, fit on train only ---
r = nb_log_count_ratio(X_train_vec, y_train)
r_sparse = sp.diags(r)

# Reweight every TF-IDF feature column by its NB log-count ratio
X_train_nb = sp.csr_matrix(X_train_vec.dot(r_sparse))
X_test_nb = sp.csr_matrix(X_test_vec.dot(r_sparse))

# --- Linear SVM trained on the NB-reweighted features ---
svm = LinearSVC(C=1.0)
svm.fit(X_train_nb, y_train)
y_pred = svm.predict(X_test_nb)
results = [evaluate_model("NB-SVM Hybrid", y_test, y_pred)]
fig_confusion_matrix("nb_svm", y_test, y_pred)

summary = pd.DataFrame(results).set_index("model")
print("\n=== Summary ===")
print(summary.round(4))

SAVE_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"vectorizer": vectorizer, "nb_log_count_ratio": r, "svm": svm},
    SAVE_DIR / "nb_svm_model.joblib",
)

export_model_metrics(results, SAVE_DIR / "nb_svm_metrics.json")
print(f"\nSaved NB-SVM bundle and metrics to {SAVE_DIR}")