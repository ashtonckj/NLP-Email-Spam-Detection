import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import clean_text_heavy, load_split

SPAM_CSV = ROOT_DIR / "data" / "processed" / "spam.csv"
SAVE_DIR = ROOT_DIR / "src" / "saved_models"
OUT_DIR = ROOT_DIR / "output" / "nb_lr_rf_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

X_train, X_test, y_train, y_test = load_split(SPAM_CSV)

X_train_text = X_train["Message"].apply(lambda t: clean_text_heavy(t, ps, stop_words))
X_test_text = X_test["Message"].apply(lambda t: clean_text_heavy(t, ps, stop_words))

CV_FOLDS = 3  
DEFAULTS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "nb_alpha": 1.0,
    "lr_C": 1.0,
    "rf_n_estimators": 200,
    "rf_max_depth": None,
    "voting_weights": (1, 1, 1),
}

MAX_FEATURES_GRID = [3000, 5000]
NGRAM_RANGE_GRID = [(1, 1), (1, 2)]
NB_ALPHA_GRID = [0.1, 0.5, 1.0]
LR_C_GRID = [0.1, 1.0, 10.0]
RF_N_ESTIMATORS_GRID = [100, 150, 200]
RF_MAX_DEPTH_GRID = [10, 20, 30]
VOTING_WEIGHTS_GRID = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}


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


# Step 1: max_features (shared TF-IDF) 
print(f"\n--- max_features sweep (LogisticRegression, ngram_range={DEFAULTS['ngram_range']}) ---")
probe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=DEFAULTS["ngram_range"])),
    ("clf", LogisticRegression(max_iter=1000)),
])
df_max_features, best_max_features = sweep(probe, "tfidf__max_features", MAX_FEATURES_GRID, label="max_features")
plot_sweep_numeric(df_max_features, "max_features", OUT_DIR / "max_features.png")
df_max_features.to_csv(OUT_DIR / "max_features.csv", index=False)

# Step 2: ngram_range (shared TF-IDF), using best_max_features from Step 1
print(f"\n--- ngram_range sweep (LogisticRegression, max_features={best_max_features}) ---")
probe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=best_max_features)),
    ("clf", LogisticRegression(max_iter=1000)),
])
df_ngram, best_ngram_range = sweep(probe, "tfidf__ngram_range", NGRAM_RANGE_GRID, label="ngram_range")
plot_sweep_categorical(df_ngram, "ngram_range", OUT_DIR / "ngram_range.png")
df_ngram.to_csv(OUT_DIR / "ngram_range.csv", index=False)

# Step 3: MultinomialNB's alpha (Laplace smoothing), TF-IDF fixed at best
print(f"\n--- nb_alpha sweep (max_features={best_max_features}, ngram_range={best_ngram_range}) ---")
pipeline_nb = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=best_max_features, ngram_range=best_ngram_range)),
    ("clf", MultinomialNB()),
])
df_alpha, best_alpha = sweep(pipeline_nb, "clf__alpha", NB_ALPHA_GRID, label="nb_alpha")
plot_sweep_numeric(df_alpha, "nb_alpha", OUT_DIR / "nb_alpha.png")
df_alpha.to_csv(OUT_DIR / "nb_alpha.csv", index=False)

# Step 4: LogisticRegression's C (inverse regularisation strength)
print(f"\n--- lr_C sweep (max_features={best_max_features}, ngram_range={best_ngram_range}) ---")
pipeline_lr = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=best_max_features, ngram_range=best_ngram_range)),
    ("clf", LogisticRegression(max_iter=1000)),
])
df_C, best_C = sweep(pipeline_lr, "clf__C", LR_C_GRID, label="lr_C", )
plot_sweep_numeric(df_C, "lr_C", OUT_DIR / "lr_C.png", logx=True)
df_C.to_csv(OUT_DIR / "lr_C.csv", index=False)

# Step 5: RandomForestClassifier's n_estimators
print(f"\n--- rf_n_estimators sweep (max_features={best_max_features}, ngram_range={best_ngram_range}) ---")
pipeline_rf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=best_max_features, ngram_range=best_ngram_range)),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=1)),
])
df_n_estimators, best_n_estimators = sweep(pipeline_rf, "clf__n_estimators", RF_N_ESTIMATORS_GRID, label="rf_n_estimators")
plot_sweep_numeric(df_n_estimators, "rf_n_estimators", OUT_DIR / "rf_n_estimators.png")
df_n_estimators.to_csv(OUT_DIR / "rf_n_estimators.csv", index=False)

# Step 6: RandomForestClassifier's max_depth, using best_n_estimators
print(f"\n--- rf_max_depth sweep (rf_n_estimators={best_n_estimators}) ---")
pipeline_rf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=best_max_features, ngram_range=best_ngram_range)),
    ("clf", RandomForestClassifier(n_estimators=best_n_estimators, random_state=42, n_jobs=1)),
])
df_max_depth, best_max_depth = sweep(pipeline_rf, "clf__max_depth", RF_MAX_DEPTH_GRID, label="rf_max_depth")
plot_sweep_numeric(df_max_depth, "rf_max_depth", OUT_DIR / "rf_max_depth.png")
df_max_depth.to_csv(OUT_DIR / "rf_max_depth.csv", index=False)

# Step 7: Soft-voting weights -- tuned as a normal VotingClassifier
# Weight order: (Naive Bayes, Logistic Regression, Random Forest).
print("\n--- voting_weights sweep ---")
pipeline_vote = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=best_max_features, ngram_range=best_ngram_range)),
    ("voting", VotingClassifier(
        estimators=[
            ("nb", MultinomialNB(alpha=best_alpha)),
            ("lr", LogisticRegression(C=best_C, max_iter=1000)),
            ("rf", RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42, n_jobs=1)),
        ],
        voting="soft",
    )),
])
df_weights, best_weights = sweep(pipeline_vote, "voting__weights", VOTING_WEIGHTS_GRID, label="voting_weights")
plot_sweep_categorical(df_weights, "voting_weights", OUT_DIR / "voting_weights.png")
df_weights.to_csv(OUT_DIR / "voting_weights.csv", index=False)

# Final tuned configuration summary
final_config = {
    "max_features": best_max_features,
    "ngram_range": best_ngram_range,
    "nb_alpha": best_alpha,
    "lr_C": best_C,
    "rf_n_estimators": best_n_estimators,
    "rf_max_depth": best_max_depth,
    "voting_weights": best_weights,
}

print("\n" + "=" * 62)
print("FINAL TUNED CONFIGURATION")
print("=" * 62)
for name, value in final_config.items():
    print(f"{name:<16} {value}")