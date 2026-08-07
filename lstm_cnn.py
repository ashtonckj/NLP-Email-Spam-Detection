import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from keras.callbacks import Callback, EarlyStopping
from keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    MaxPooling1D,
)
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Navigate up 2 levels from testing.py to find the project root directory
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.preprocessing.preprocessing import clean_text


class GuiLogger(Callback):

    def __init__(self, log_callback, progress_callback=None):
        super().__init__()
        self.log_callback = log_callback
        self.progress_callback = progress_callback

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.log_callback(
            f"Epoch {epoch+1} completed | "
            f"Loss: {logs.get('loss'):.4f} | "
            f"Accuracy: {logs.get('accuracy'):.4f} | "
            f"Val Accuracy: {logs.get('val_accuracy'):.4f}"
        )
        if self.progress_callback:
            self.progress_callback(epoch + 1)

def train_lstm_cnn(filepath, log_callback=print, progress_callback=None):
    data = pd.read_csv(filepath)

    print(data.head())
    print(data.info())
    print(data["Category"].value_counts())

    # Keep required columns
    data = data[["Message", "Category"]]

    # Encode category
    # encoder = LabelEncoder()
    # data["Category"] = encoder.fit_transform(data["Category"])
    data["Message"] = data["Message"].apply(clean_text)

    X = data["Message"]
    y = data["Category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Tokenization
    vocab_size = 10000
    tokenizer = Tokenizer(
        num_words=vocab_size
    )
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(
        X_train
    )
    X_test_seq = tokenizer.texts_to_sequences(
        X_test
    )

    # Padding
    max_length = 200
    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=max_length,
        padding="post"
    )
    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=max_length,
        padding="post"
    )

    model = Sequential()
    
    # 1. Embedding Layer: Converts integer sequences into dense word vectors
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=128
        )
    )
    
    # 2. CNN Layers: Extract local features (e.g., spammy n-grams or keyword combinations)
    model.add(
        Conv1D(
            filters=64, 
            kernel_size=5, 
            activation="relu"
        )
    )
    model.add(
        MaxPooling1D(
            pool_size=4
        )
    )
    
    # 3. LSTM Layer: Learns long-term sequential dependencies from the CNN's feature maps
    model.add(
        LSTM(
            128,
            dropout=0.2,
            recurrent_dropout=0.2
        )
    )
    
    # 4. Dense/Classification Layers
    model.add(
        Dense(
            64,
            activation="relu"
        )
    )
    model.add(
        Dropout(0.5)
    )
    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    log_callback("Training started... Please wait...")
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    model.fit(
        X_train_pad,
        y_train,
        epochs=10,
        batch_size=32, 
        validation_split=0.2,
        callbacks=[
            GuiLogger(log_callback, progress_callback),
            early_stop
        ]
    )

    pred = model.predict(
        X_test_pad
    )
    pred = (
        pred > 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        pred
    )
    print("Accuracy:", accuracy)
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))

    result = {
        "accuracy": accuracy,
        "report": classification_report(y_test, pred)
    }

    # Create folder if it doesn't exist
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    # Save the deep learning model architecture and weights
    model.save("saved_models/lstm_cnn_model.keras")
    
    # Save the tokenizer (you need this to process the user's text later!)
    with open('saved_models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the accuracy so the GUI can display it
    metrics = {"accuracy": f"{accuracy * 100:.2f}%"}
    with open('saved_models/lstm_cnn_metrics.json', 'w') as f:
        json.dump(metrics, f)

    return result

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "data", "Ceas08_Enron.csv")
    train_lstm_cnn(path)