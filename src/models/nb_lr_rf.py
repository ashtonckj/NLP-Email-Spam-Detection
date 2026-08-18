# Hybrid Model: Soft-voting ensemble Naive Bayes, Logistic Regression, and Random Forest
import sys
from pathlib import Path

import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.figures.figures_data_analysis import fig_confusion_matrix
from src.models.evaluation import evaluate_model, export_model_metrics
from src.preprocessing.preprocessing import clean_text_heavy, load_split

SPAM_CSV = ROOT_DIR / "data" / "processed" / "spam.csv"
SAVE_DIR = ROOT_DIR / "src" / "saved_models"

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
fig_confusion_matrix("nb_lr_rf", y_test, y_pred)

summary = pd.DataFrame(results).set_index("model")
print("\n=== Summary ===")
print(summary.round(4))

SAVE_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"hybrid": hybrid, "vectorizer": vectorizer},
    SAVE_DIR / "nb_lr_rf_model.joblib",
)

export_model_metrics(results, SAVE_DIR / "nb_lr_rf_metrics.json")
print(f"\nSaved NB+LR+RF bundle and metrics to {SAVE_DIR}")