# Hybrid Model: LSTM + CNN
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
from sklearn.metrics import classification_report

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.figures.figures_data_analysis import fig_confusion_matrix
from src.models.evaluation import evaluate_model, export_model_metrics
from src.preprocessing.preprocessing import load_split

SPAM_CSV = ROOT_DIR / "data" / "processed" / "spam.csv"
SAVE_DIR = ROOT_DIR / "src" / "saved_models"


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
result = evaluate_model("LSTM + CNN Hybrid", y_test, y_pred)
fig_confusion_matrix("lstm_cnn", y_test, y_pred)

print("\nDetailed classification report:")
print(classification_report(y_test, y_pred))

SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save(SAVE_DIR / "lstm_cnn_model.keras")

with open(SAVE_DIR / "lstm_cnn_tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

export_model_metrics(result, SAVE_DIR / "lstm_cnn_metrics.json")
print(f"\nSaved LSTM-CNN model and metrics to {SAVE_DIR}")