# 📧 Spam Email Detection

A machine learning project that classifies email messages as spam or ham. It trains and compares three hybrid models — an LSTM + CNN neural network, a soft-voting ensemble of Naive Bayes / Logistic Regression / Random Forest, and an NB-reweighted linear SVM — and lets you test them side by side through a desktop GUI.

## Requirements

Python 3.10+ is recommended (Keras 3's `TextVectorization` layer is used). Install the packages used across the project:

```bash
pip install pandas numpy scikit-learn scipy joblib nltk tensorflow keras matplotlib seaborn
```

The TF-IDF based models (NB+LR+RF, NB+SVM) need NLTK's stopwords corpus. It's downloaded automatically the first time it's needed, but you can also grab it upfront:

```bash
python -m nltk.downloader stopwords
```

## Running the app

From the project root:

```bash
python app.py
```

Paste (or type) an email's text into the box and hit **Analyze Email** — it runs the text through every model found in `src/saved_models/` and shows each one's prediction, confidence, and reported accuracy side by side.

---

## Models included

| Model | Training script | Saved as |
|---|---|---|
| LSTM + CNN Hybrid | `src/models/lstm_cnn.py` | `lstm_cnn_model.keras` + `lstm_cnn_tokenizer.pkl` |
| NB + LR + RF Hybrid | `src/models/nb_lr_rf.py` | `nb_lr_rf_model.joblib` |
| NB + SVM Hybrid | `src/models/nb_svm.py` | `nb_svm_model.joblib` |

Each training script loads the same train/test split (`load_split`), evaluates its model(s) with `evaluate_model`, and exports metrics via `export_model_metrics` — both shared from `src/models/evaluation.py`. Run any script directly (e.g. `python src/models/nb_svm.py`) to retrain and re-save that model.

The GUI (`app.py`) doesn't scan `saved_models/` for files — it reads a hardcoded `MODEL_REGISTRY` at the top of the file. If you rename or add a model, update that list to match.

## Project structure

```
project-root/
├── app.py                          # Tkinter GUI — run from here to test models
├── data/
│   ├── processed/
│   │   └── spam.csv                # cleaned dataset used for training
│   └── raw/
│       ├── CEAS_08.csv             # raw dataset before preprocessing
│       ├── Enron.csv               # raw dataset before preprocessing
│       └── Nazario.csv             # raw dataset before preprocessing
├── src/
│   ├── preprocessing/
│   │   └── preprocessing.py        # clean_text_light / clean_text_heavy / load_split
│   ├── figures/
│   │   └── figures_data_analysis.py     # figures used in documentation
│   │   └── figures_datasets_info.ipynb  # processed/raw datasets .info()
│   │   └── figures_wordcloud.py         # wordcloud generator
│   ├── models/
│   │   ├── evaluation.py           # evaluate_model() + export_model_metrics()
│   │   ├── lstm_cnn.py             # LSTM + CNN hybrid
│   │   ├── nb_lr_rf.py             # NB + LR + RF soft-voting ensemble
│   │   └── nb_svm.py               # NB log-count-ratio reweighted linear SVM
│   └── saved_models/               # holds trained models + tokenizers + metrics
│       ├── lstm_cnn_model.keras
│       ├── lstm_cnn_tokenizer.pkl
│       ├── lstm_cnn_metrics.json
│       ├── nb_lr_rf_model.joblib
│       ├── nb_lr_rf_metrics.json
│       ├── nb_svm_model.joblib
│       └── nb_svm_metrics.json
└── README.md
```