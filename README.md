# 📱 SMS Spam Detection By Naive Bayes

A complete end-to-end project that classifies SMS messages as **Spam** or **Ham (legitimate)** using text preprocessing, exploratory data analysis, and multiple Naive Bayes classifier variants.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [EDA Insights](#eda-insights)
- [Text Preprocessing](#text-preprocessing)
- [Model Results](#model-results)
- [Key Takeaways](#key-takeaways)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

---

## Overview

Spam detection is a classic and highly practical NLP classification problem. This project builds a spam filter from scratch — starting from raw, messy CSV data all the way to trained, evaluated classifiers — without relying on any pre-built spam detection library.

The core idea: transform raw SMS text into numerical features using a **Bag of Words** (CountVectorizer) approach, then compare three variants of the **Naive Bayes** algorithm to find the best model for this task.

---

## Dataset

| Property | Value |
|---|---|
| File | `spam.csv` |
| Raw records | 5,572 |
| After removing duplicates | **5,169** |
| Ham messages | **4,516 (87.4%)** |
| Spam messages | **653 (12.6%)** |
| Missing values | None |

The dataset is heavily **imbalanced** — spam makes up only ~12.6% of all messages. This makes **precision** a critical metric: a poor model could simply predict "ham" every time and still achieve ~87% accuracy. This is why raw accuracy alone is not a reliable indicator of model quality here.

---

## Project Pipeline

```
Raw CSV  →  Data Cleaning  →  EDA  →  Text Preprocessing  →  Vectorization  →  Model Training  →  Evaluation
```

### 1. Data Cleaning
- Dropped 3 unnamed, mostly-empty columns (`Unnamed: 2/3/4`)
- Renamed `v1 → target` and `v2 → text` for clarity
- Label-encoded the target column: `ham = 0`, `spam = 1`
- Removed **403 duplicate entries** (5,572 → 5,169 records)

### 2. Feature Engineering
Three new statistical features were derived from each message:

| Feature | Description |
|---|---|
| `num_characters` | Total character count of the message |
| `num_words` | Total word count (whitespace-split) |
| `num_sentences` | Number of sentences (NLTK sentence tokenizer) |

### 3. Text Preprocessing
Each SMS message was cleaned through a 5-step NLP pipeline before vectorization:

```
Raw Text
   ↓  Lowercase
   ↓  Tokenize (NLTK word_tokenize)
   ↓  Remove special characters (keep alphanumeric only)
   ↓  Remove stop words & punctuation
   ↓  Porter Stemming
   →  Clean Token String
```

### 4. Vectorization
The cleaned text was converted to a numerical matrix using **CountVectorizer** (Bag of Words):
- **Vocabulary size: 9,524 unique tokens**
- Each message becomes a vector of token frequencies

### 5. Train / Test Split
| Split | Size |
|---|---|
| Training set | 4,135 messages (80%) |
| Test set | 1,034 messages (20%) |
| Random state | 2 |

---

## EDA Insights

### Class Imbalance
The dataset is skewed — 87.4% of messages are ham. Any classifier must beat this baseline to be useful.

### Message Length Patterns

| Metric | Ham (avg) | Spam (avg) |
|---|---|---|
| Characters | **70.5** | **137.9** |
| Words | **14.1** | **23.7** |

> **Key insight:** Spam messages are nearly **2× longer** than ham messages on average. Length alone is a meaningful signal.

Ham messages also have much higher variance in length (std dev: 56 chars) compared to spam (std dev: 30 chars) — spam messages cluster tightly around ~137–157 characters, suggesting they often follow templates.

### Word Frequency Patterns
- **Spam corpus** is dominated by words like: *free*, *call*, *claim*, *prize*, *win*, *text*, *reply*, *urgent*
- **Ham corpus** is dominated by everyday conversational words: *go*, *get*, *come*, *ok*, *know*, *like*, *good*

These distinctly different vocabularies are exactly why Naive Bayes (which models word probabilities per class) works so well here.

---

## Text Preprocessing

The custom `transform_text()` function processes each message step by step:

```python
def transform_text(text):
    text = text.lower()                          # Lowercase
    text = nltk.word_tokenize(text)              # Tokenize
    text = [i for i in text if i.isalnum()]      # Remove special chars
    text = [i for i in text                      # Remove stopwords
            if i not in stopwords.words('english')
            and i not in string.punctuation]
    text = [ps.stem(i) for i in text]            # Stem words
    return " ".join(text)
```

**Why stemming?** Words like "calling", "called", "calls" all stem to "call" — reducing vocabulary noise and helping the model generalize better.

---

## Model Results

Three Naive Bayes variants were trained and evaluated on the same test set (1,034 messages):

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Gaussian Naive Bayes | 91.20% | 61.58% | 90.58% | 73.31% |
| Multinomial Naive Bayes | 96.52% | 82.28% | 94.20% | 87.84% |
| **Bernoulli Naive Bayes** | **97.10%** | **98.21%** | 79.71% | **88.00%** |

### Confusion Matrix Breakdown

**Gaussian NB**
```
              Predicted Ham   Predicted Spam
Actual Ham        818              78        ← 78 false alarms
Actual Spam        13             125
```

**Multinomial NB**
```
              Predicted Ham   Predicted Spam
Actual Ham        868              28
Actual Spam         8             130        ← misses 8 spam
```

**Bernoulli NB**
```
              Predicted Ham   Predicted Spam
Actual Ham        894               2        ← only 2 false alarms
Actual Spam        28             110
```

### Which model to choose?

| Goal | Best Model |
|---|---|
| Minimize false positives (legitimate SMS flagged as spam) | **Bernoulli NB** (Precision: 98.21%) |
| Catch as many spam messages as possible | **Multinomial NB** (Recall: 94.20%) |
| Balanced performance | **Bernoulli NB** (F1: 88.00%) |

> **Recommendation: Bernoulli NB** is the best overall choice. With a precision of **98.21%**, it almost never flags a legitimate message as spam — which is critical for a real-world SMS filter where false positives directly harm the user experience. It only misses ~28 spam messages out of 138 in the test set, which is an acceptable trade-off.

**Why does Gaussian NB perform worst?** Gaussian NB assumes features follow a normal (Gaussian) distribution. Word count vectors are discrete and sparse — far from Gaussian — making this assumption a poor fit for text data.

---

## Key Takeaways

1. **Accuracy is misleading on imbalanced data.** Gaussian NB has 91.2% accuracy but only 61.6% precision — it would flood users with false spam alerts.
2. **Spam messages are structurally different** — longer, denser, more templated, and vocabulary-distinct from ham.
3. **Bernoulli NB excels at precision** because it treats features as binary (word present/absent), making it robust against noisy frequency counts in short texts.
4. **Stemming + stop word removal** significantly reduces vocabulary noise, improving generalization.
5. **Bag of Words is surprisingly powerful** for this task despite ignoring word order and context — a testament to how distinctive spam vocabulary is.

---

## Tech Stack

| Category | Library | Purpose |
|---|---|---|
| Data Handling | `pandas`, `numpy` | Loading, cleaning, feature engineering |
| Visualization | `matplotlib`, `seaborn` | EDA plots, heatmaps, histograms |
| Text Visualization | `wordcloud` | Word cloud generation for spam/ham |
| NLP | `nltk` | Tokenization, stopwords, stemming |
| ML | `scikit-learn` | Vectorization, models, metrics |

**Python version:** 3.x

---

## Project Structure

```
sms-spam-detection/
│
├── sms_spam_detection.ipynb    # Main Jupyter notebook (full pipeline)
├── spam.csv                    # Raw dataset
└── README.md                   # Project documentation
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud
```

### 3. Download NLTK resources
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### 4. Launch the notebook
```bash
jupyter notebook sms_spam_detection.ipynb
```

Run all cells top to bottom. The notebook is self-contained and follows the pipeline in order: cleaning → EDA → preprocessing → modeling → evaluation.

---

## Possible Improvements

- **TF-IDF Vectorizer** instead of CountVectorizer to down-weight common words
- **Bigrams/Trigrams** to capture phrases like "free prize" or "call now"
- **Class balancing** using SMOTE or class weights to handle the 87/13 imbalance
- **Other classifiers** — Logistic Regression, SVM, Random Forest for comparison
- **Hyperparameter tuning** using GridSearchCV
- **Deployment** as a simple web app using Streamlit or Flask

---

## License

This project is open-source and available under the [MIT License](LICENSE).
