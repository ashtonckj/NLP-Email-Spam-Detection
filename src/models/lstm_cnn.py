# Train LSTM+CNN hybrid model on the shared spam.csv dataset
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    MaxPooling1D,
    TextVectorization,
)
from keras.models import Sequential
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.figures.figures_data_analysis import fig_confusion_matrix
from src.preprocessing.preprocessing import load_split

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


# Same split every model uses -- Message column already light-cleaned in the shared pipeline
X_train, X_test, y_train, y_test = load_split(SPAM_CSV)
X_train_text = X_train["Message"]
X_test_text = X_test["Message"]

# Tokenization
vocab_size = 10000
max_length = 200
tokenizer = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=max_length,
)
tokenizer.adapt(np.array(X_train_text))

X_train_pad = tokenizer(np.array(X_train_text)).numpy()
X_test_pad = tokenizer(np.array(X_test_text)).numpy()

model = Sequential()

# 1. Embedding layer: converts integer sequences into dense word vectors
model.add(Embedding(input_dim=vocab_size, output_dim=128))

# 2. CNN layers: extract local features (e.g. spammy n-grams or keyword combos)
model.add(Conv1D(filters=64, kernel_size=5, activation="relu"))
model.add(MaxPooling1D(pool_size=4))

# 3. LSTM layer: learns long-term sequential dependencies from the CNN's feature maps
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

# 4. Dense/classification layers
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

print("Training started...")
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

model.fit(
    X_train_pad,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
)

pred_prob = model.predict(X_test_pad)
y_pred = (pred_prob > 0.5).astype(int)
result = evaluate_model("LSTM + CNN", y_test, y_pred)
fig_confusion_matrix(y_test, y_pred, "lstm_cnn")

print("\nDetailed classification report:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model, tokenizer, and metrics
SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save(SAVE_DIR / "lstm_cnn_model.keras")

with open(SAVE_DIR / "tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(SAVE_DIR / "lstm_cnn_metrics.json", "w") as f:
    json.dump({"accuracy": f"{result['accuracy'] * 100:.2f}%"}, f)
