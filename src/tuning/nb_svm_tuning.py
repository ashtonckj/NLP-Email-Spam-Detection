import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import clean_text_heavy, load_split

SPAM_CSV = ROOT_DIR / "data" / "processed" / "spam.csv"
OUT_DIR = ROOT_DIR / "output" / "nb_svm_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

X_train, X_test, y_train, y_test = load_split(SPAM_CSV)

X_train_text = X_train["Message"].apply(lambda t: clean_text_heavy(t, ps, stop_words))

CV_FOLDS = 3

DEFAULTS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "nb_alpha": 1.0,
    "svm_C": 1.0,
}

MAX_FEATURES_GRID = [3000, 5000]
NGRAM_RANGE_GRID = [(1, 1), (1, 2)]
NB_ALPHA_GRID = [0.1, 0.5, 1.0]
SVM_C_GRID = [0.1, 1.0, 10.0]

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}


class NBFeatureScaler(BaseEstimator, TransformerMixin):
    """Reweights TF-IDF columns by their Naive Bayes log-count ratio r --
    same formula as nb_svm.py's nb_log_count_ratio() -- so LinearSVC pays
    more attention to terms whose weight is disproportionately concentrated
    in one class. Wrapped as a transformer so it can sit inside a Pipeline
    and have its alpha tuned by GridSearchCV like any other hyperparameter.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = sp.csr_matrix(X)
        y = np.asarray(y)
        p = self.alpha + X[y == 1].sum(axis=0)
        q = self.alpha + X[y == 0].sum(axis=0)
        p = np.asarray(p).ravel()
        q = np.asarray(q).ravel()
        self.r_ = np.log((p / p.sum()) / (q / q.sum()))
        return self

    def transform(self, X):
        r_sparse = sp.diags(self.r_)
        return sp.csr_matrix(sp.csr_matrix(X).dot(r_sparse))


def sweep(estimator, param_name, values, label=None, cv=CV_FOLDS):
    label = label or param_name
    grid = GridSearchCV(
        estimator=estimator,
        param_grid={param_name: values},
        scoring=SCORING,
        refit="f1",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train_text, y_train)
    cv_results = grid.cv_results_

    rows = []
    for i, v in enumerate(values):
        rows.append({
            label: v,
            "Accuracy": cv_results["mean_test_accuracy"][i],
            "Precision": cv_results["mean_test_precision"][i],
            "Recall": cv_results["mean_test_recall"][i],
            "F1": cv_results["mean_test_f1"][i],
            "Time(s)": cv_results["mean_fit_time"][i],
        })
        print(f"  {label}={v}: Acc={rows[-1]['Accuracy']:.4f} "
              f"P={rows[-1]['Precision']:.4f} R={rows[-1]['Recall']:.4f} "
              f"F1={rows[-1]['F1']:.4f} avg_fit_time={rows[-1]['Time(s)']:.3f}s")

    best_value = grid.best_params_[param_name]
    print(f"  -> GridSearchCV best {label} = {best_value} (F1={grid.best_score_:.4f})")
    return pd.DataFrame(rows), best_value


def plot_sweep_numeric(df, label, out_path, xlabel=None, logx=False):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = df[label]
    ax1.plot(x, df["F1"], marker="o", color="#2563eb", label="F1 (mean CV)")
    ax1.set_xlabel(xlabel or label)
    ax1.set_ylabel("F1 score", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    if logx:
        ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.plot(x, df["Time(s)"], marker="s", color="#dc2626", linestyle="--", label="Avg fit time (s)")
    ax2.set_ylabel("Avg fit time per model (s)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    plt.title(f"Effect of {label} on F1 and Fit Time")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  saved", out_path)


def plot_sweep_categorical(df, label, out_path):
    plt.figure(figsize=(6, 4))
    plt.bar([str(v) for v in df[label]], df["F1"], color="#2563eb")
    plt.ylabel("F1 score (mean CV)")
    plt.xlabel(label)
    plt.title(f"Effect of {label} on F1")
    plt.ylim(max(0, df["F1"].min() - 0.01), 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  saved", out_path)


# Step 1: max_features (TF-IDF)
print(f"\n--- max_features sweep (nb_alpha={DEFAULTS['nb_alpha']}, svm_C={DEFAULTS['svm_C']}) ---")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=DEFAULTS["ngram_range"])),
    ("nb_scale", NBFeatureScaler(alpha=DEFAULTS["nb_alpha"])),
    ("svm", LinearSVC(C=DEFAULTS["svm_C"])),
])
df_max_features, best_max_features = sweep(pipeline, "tfidf__max_features", MAX_FEATURES_GRID, label="max_features")
plot_sweep_numeric(df_max_features, "max_features", OUT_DIR / "01_max_features.png")
df_max_features.to_csv(OUT_DIR / "01_max_features.csv", index=False)

# Step 2: ngram_range (TF-IDF), using best_max_features from Step 1
print(f"\n--- ngram_range sweep (max_features={best_max_features}) ---")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=best_max_features)),
    ("nb_scale", NBFeatureScaler(alpha=DEFAULTS["nb_alpha"])),
    ("svm", LinearSVC(C=DEFAULTS["svm_C"])),
])
df_ngram, best_ngram_range = sweep(pipeline, "tfidf__ngram_range", NGRAM_RANGE_GRID, label="ngram_range")
plot_sweep_categorical(df_ngram, "ngram_range", OUT_DIR / "02_ngram_range.png")
df_ngram.to_csv(OUT_DIR / "02_ngram_range.csv", index=False)

# Step 3: NB log-count ratio's alpha (smoothing), TF-IDF fixed at best
print(f"\n--- nb_alpha sweep (max_features={best_max_features}, ngram_range={best_ngram_range}) ---")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=best_max_features, ngram_range=best_ngram_range)),
    ("nb_scale", NBFeatureScaler()),
    ("svm", LinearSVC(C=DEFAULTS["svm_C"])),
])
df_alpha, best_alpha = sweep(pipeline, "nb_scale__alpha", NB_ALPHA_GRID, label="nb_alpha")
plot_sweep_numeric(df_alpha, "nb_alpha", OUT_DIR / "03_nb_alpha.png")
df_alpha.to_csv(OUT_DIR / "03_nb_alpha.csv", index=False)

# Step 4: LinearSVC's C (regularisation strength), using best_alpha
print(f"\n--- svm_C sweep (max_features={best_max_features}, ngram_range={best_ngram_range}, nb_alpha={best_alpha}) ---")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=best_max_features, ngram_range=best_ngram_range)),
    ("nb_scale", NBFeatureScaler(alpha=best_alpha)),
    ("svm", LinearSVC()),
])
df_C, best_C = sweep(pipeline, "svm__C", SVM_C_GRID, label="svm_C")
plot_sweep_numeric(df_C, "svm_C", OUT_DIR / "04_svm_C.png", logx=True)
df_C.to_csv(OUT_DIR / "04_svm_C.csv", index=False)

# Final tuned configuration summary
final_config = {
    "max_features": best_max_features,
    "ngram_range": best_ngram_range,
    "nb_alpha": best_alpha,
    "svm_C": best_C,
}

print("\n" + "=" * 62)
print("FINAL TUNED CONFIGURATION")
print("=" * 62)
for name, value in final_config.items():
    print(f"{name:<16} {value}")