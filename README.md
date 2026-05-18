# DreamTeam-Internship-Assg02-SMS-Spam-Prediction-
# SMS Spam Detection 📩

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using Natural Language Processing (NLP) techniques and Naive Bayes classifiers.

---

## 📁 Dataset

- **File:** `spam.csv`
- **Source:** SMS Spam Collection Dataset
- **Total Messages:** 5,572
  - Ham (legitimate): 4,825 (~86.6%)
  - Spam: 747 (~13.4%)
- **Features used:** Message text (`text`) and label (`target`)

---

## 🔄 Project Workflow

### 1. Data Cleaning
- Dropped irrelevant unnamed columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`)
- Renamed columns: `v1 → target`, `v2 → text`
- Encoded target labels using `LabelEncoder` (ham = 0, spam = 1)
- Removed duplicate entries

### 2. Exploratory Data Analysis (EDA)
- Visualized class distribution via pie chart
- Engineered features: number of characters, words, and sentences per message
- Compared feature distributions between ham and spam messages using histograms
- Generated pairplots and a correlation heatmap
- Key insight: Spam messages tend to be significantly longer than ham messages

### 3. Text Preprocessing
Each message goes through the following pipeline:
- Lowercasing
- Tokenization (via NLTK)
- Removal of special characters (kept alphanumeric only)
- Stop word and punctuation removal
- Porter Stemming

### 4. Visualization
- Word clouds generated for both spam and ham messages
- Top 30 most frequent words plotted for spam and ham corpora separately

### 5. Model Building
- Text vectorized using **CountVectorizer** (Bag of Words)
- 80/20 train-test split (`random_state=2`)
- Three Naive Bayes variants trained and evaluated:

| Model | Notes |
|---|---|
| Gaussian Naive Bayes | Baseline |
| Multinomial Naive Bayes | Suited for word count features |
| Bernoulli Naive Bayes | Suited for binary presence/absence |

**Metrics evaluated:** Accuracy, Confusion Matrix, Precision, Recall, F1 Score

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| NLP | `nltk` (tokenization, stopwords, stemming) |
| ML | `scikit-learn` (CountVectorizer, Naive Bayes, metrics) |

---

## 🚀 How to Run

1. **Clone the repository**
```bash
   git clone <your-repo-url>
   cd <repo-folder>
```

2. **Install dependencies**
```bash
   pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud
```

3. **Download NLTK data** (runs automatically in the notebook, or manually):
```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
```

4. **Place the dataset** — ensure `spam.csv` is in the same directory as the notebook.

5. **Run the notebook**
```bash
   jupyter notebook sms_spam_detection.ipynb
```

---

## 📊 Results

Three Naive Bayes models were compared. **Multinomial NB** and **Bernoulli NB** are generally best suited for text classification with bag-of-words features and are expected to outperform Gaussian NB on this task. Precision is prioritized to minimize false positives (legitimate messages flagged as spam).

---

## 📂 File Structure

```
├── sms_spam_detection.ipynb   # Main notebook
├── spam.csv                   # Dataset
└── README.md
```
