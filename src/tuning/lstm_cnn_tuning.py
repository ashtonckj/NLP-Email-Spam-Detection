import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import load_split

SPAM_CSV = ROOT_DIR / "data" / "processed" / "spam.csv"
OUT_DIR = ROOT_DIR / "output" / "lstm_cnn_tuning"

SELECTION_METRIC = "f1"

VALIDATION_SIZE = 0.20
RANDOM_SEED = 42
QUICK = "--quick" in sys.argv

BASELINE = {
    "vocab_size": 10000,
    "max_length": 200,
    "embedding_dim": 128,
    "conv_filters": 64,
    "kernel_size": 5,
    "pool_size": 4,
    "lstm_units": 128,
    "lstm_dropout": 0.2,
    "recurrent_dropout": 0.0,
    "dense_units": 64,
    "dense_dropout": 0.5,
    "optimizer": "adam",
    "batch_size": 32,
}

SWEEPS = {
    "vocab_size":        [2000, 5000, 10000],
    "max_length":        [100, 150, 200],
    "embedding_dim":     [64, 128],
    "conv_filters":      [64, 256],
    "kernel_size":       [5, 7],
    "pool_size":         [4, 8],
    "lstm_units":        [64, 128, 256],
    "lstm_dropout":      [0.0, 0.2, 0.4],
    "recurrent_dropout": [0.0],
    "dense_units":       [32, 64, 128],
    "dense_dropout":     [0.3, 0.5, 0.7],
    "optimizer":         ["adam", "rmsprop", "sgd"],
    "batch_size":        [16, 32, 64],
}

TEXT_PARAMS = {"vocab_size", "max_length"}

PALETTE = {"bar": "#4C72B0", "best": "#55A868", "edge": "#22303F"}


def get_train_validation_split():
    X_train_full, _X_test, y_train_full, _y_test = load_split(SPAM_CSV)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=y_train_full,
    )
    return (np.array(X_tr["Message"]), np.array(X_val["Message"]),
            np.asarray(y_tr), np.asarray(y_val))


_vec_cache = {}


def vectorise(tr_text, val_text, vocab_size, max_length):
    from keras.layers import TextVectorization

    key = (vocab_size, max_length)
    if key not in _vec_cache:
        tokenizer = TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
            output_sequence_length=max_length,
        )
        tokenizer.adapt(tr_text)
        _vec_cache[key] = (tokenizer(tr_text).numpy(), tokenizer(val_text).numpy())
    return _vec_cache[key]

def build_model(cfg):
    from keras.layers import LSTM, Conv1D, Dense, Dropout, Embedding, MaxPooling1D
    from keras.models import Sequential

    model = Sequential()
    model.add(Embedding(input_dim=cfg["vocab_size"], output_dim=cfg["embedding_dim"]))
    model.add(Conv1D(filters=cfg["conv_filters"], kernel_size=cfg["kernel_size"],
                     activation="relu"))
    model.add(MaxPooling1D(pool_size=cfg["pool_size"]))
    model.add(LSTM(cfg["lstm_units"],
                   dropout=cfg["lstm_dropout"],
                   recurrent_dropout=cfg["recurrent_dropout"]))
    model.add(Dense(cfg["dense_units"], activation="relu"))
    model.add(Dropout(cfg["dense_dropout"]))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer=cfg["optimizer"],
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def run_trial(cfg, tr_text, val_text, y_tr, y_val):
    from keras.callbacks import EarlyStopping

    X_tr_pad, X_val_pad = vectorise(tr_text, val_text,
                                    cfg["vocab_size"], cfg["max_length"])
    model = build_model(cfg)

    model.fit(
        X_tr_pad, y_tr,
        epochs=3 if QUICK else 6,
        batch_size=cfg["batch_size"],
        validation_split=0.2,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2,
                                 restore_best_weights=True)],
    )

    y_pred = (model.predict(X_val_pad, verbose=0) > 0.5).astype(int)

    from keras import backend as K
    K.clear_session()   # free graph memory between trials

    return {
        "accuracy":  accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall":    recall_score(y_val, y_pred, zero_division=0),
        "f1":        f1_score(y_val, y_pred, zero_division=0),
    }


def plot_sweep(param, trials, best_value):
    labels = [str(t[param]) for t in trials]
    scores = [t[SELECTION_METRIC] * 100 for t in trials]
    colors = [PALETTE["best"] if str(t[param]) == str(best_value)
              else PALETTE["bar"] for t in trials]

    fig, ax = plt.subplots(figsize=(5.6, 4))
    bars = ax.bar(labels, scores, color=colors,
                  edgecolor=PALETTE["edge"], linewidth=0.7)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.03,
                f"{score:.2f}", ha="center", fontsize=9)

    low, high = min(scores), max(scores)
    pad = max((high - low) * 0.4, 0.15)
    ax.set_ylim(low - pad, high + pad)

    ax.set_xlabel(param)
    ax.set_ylabel(f"Validation {SELECTION_METRIC.upper()} (%)")
    ax.set_title(f"Effect of {param} on validation {SELECTION_METRIC.upper()}")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_tuning_{param}.png", dpi=300)
    plt.close(fig)


def tune_lstm_cnn():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tr_text, val_text, y_tr, y_val = get_train_validation_split()
    print(f"Training rows: {len(tr_text):,}   Validation rows: {len(val_text):,}")

    total = sum(len(v) for v in SWEEPS.values())
    print(f"Sweeping {len(SWEEPS)} parameters, {total} trials total."
          f"{'  (quick mode)' if QUICK else ''}\n")

    cfg = dict(BASELINE)
    all_trials = []
    history = []
    trial_no = 0

    for param, candidates in SWEEPS.items():
        print(f"--- Sweeping {param} over {candidates} ---")
        sweep_trials = []

        for value in candidates:
            trial_no += 1
            trial_cfg = dict(cfg)
            trial_cfg[param] = value

            start = time.time()
            scores = run_trial(trial_cfg, tr_text, val_text, y_tr, y_val)
            elapsed = round(time.time() - start, 1)

            row = {"parameter": param, param: value, **scores,
                   "seconds": elapsed, **{f"cfg_{k}": v for k, v in trial_cfg.items()}}
            sweep_trials.append(row)
            all_trials.append(row)

            print(f"  [{trial_no}/{total}] {param}={value:<8} "
                  f"{SELECTION_METRIC}={scores[SELECTION_METRIC]:.4f}  ({elapsed}s)")

        best = max(sweep_trials, key=lambda t: t[SELECTION_METRIC])
        best_value = best[param]

        cfg[param] = best_value
        if param in TEXT_PARAMS:
            _vec_cache.clear()

        history.append({"parameter": param, "best_value": best_value,
                        "best_score": best[SELECTION_METRIC]})
        plot_sweep(param, sweep_trials, best_value)

        print(f"  -> best {param} = {best_value} "
              f"({SELECTION_METRIC}={best[SELECTION_METRIC]:.4f})\n")

    pd.DataFrame(all_trials).to_csv(OUT_DIR / "tuning_all_trials.csv", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / "tuning_summary.csv", index=False)

    with open(OUT_DIR / "best_params.json", "w") as f:
        json.dump({"selection_metric": SELECTION_METRIC,
                   "method": "one-factor-at-a-time",
                   "baseline": BASELINE,
                   "best_params": cfg,
                   "final_validation_score": history[-1]["best_score"]}, f, indent=2)

    print("=" * 62)
    print("FINAL TUNED CONFIGURATION")
    print("=" * 62)
    for k, v in cfg.items():
        marker = "  <- changed" if v != BASELINE[k] else ""
        print(f"  {k:<20} {v}{marker}")
    print(f"\nValidation {SELECTION_METRIC}: {history[-1]['best_score']:.4f}")
    print(f"\nCharts, trial tables and best_params.json saved to {OUT_DIR}")


if __name__ == "__main__":
    tune_lstm_cnn()
