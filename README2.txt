# Project Title: Drug Review Sentiment Classification using DistilBERT and Logistic Regression

## Objective:
This project aims to analyze and classify **drug reviews** based on user feedback. The main goal is to predict **user sentiment or satisfaction level** using Natural Language Processing (NLP) techniques. Two types of models were used:
1. Traditional machine learning models (Logistic Regression,SVM,XGBoost Classifier)
2. A deep learning transformer model (DistilBERT)

---

## Step-by-Step Workflow:

### 🔹 Step 1: Importing Required Libraries
All essential Python libraries were imported including:
- **pandas** for data handling
- **torch** and **transformers** for deep learning
- **sklearn** for traditional ML and evaluation
- **matplotlib**, **seaborn**, and **plotly** for visualizations

---

### 🔹 Step 2: Mount Google Drive
Since the dataset is stored in Google Drive, we mounted it using:
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 🔹 Step 3: Load the Dataset
The drug review dataset (`drug_review_test.csv`) was loaded using pandas.

```python
df = pd.read_csv('/content/drive/MyDrive/.../drug_review_test.csv')
```

---

### 🔹 Step 4: Exploratory Data Analysis (EDA)

1. **Top 20 Drugs with 10/10 Rating**  
   A bar plot was created using Plotly to visualize which drugs received the highest perfect scores.

2. **Rating Distribution**  
   A pie chart was generated to show the proportion of different user ratings.

These visualizations helped understand data imbalance and popular drugs.

---

### 🔹 Step 5: Text Preprocessing
The review texts were cleaned to prepare them for modeling. This includes:
- Removing special characters
- Lowercasing
- Tokenizing

---

### 🔹 Step 6: Feature Extraction using TF-IDF (for Logistic Regression)
- The cleaned reviews were transformed into numerical form using `TfidfVectorizer`.
- This helps in converting textual data into features that machine learning models can understand.

---

### 🔹 Step 7: Models are — Logistic Regression, SVM and XGBoost Classifier

1. The data was split into training and test sets.
2. A **Logistic Regression** model was trained using the TF-IDF features.
3. A **SVM** model was trained using scikit-learn’s LinearSVC.
4. A **XGBoost Classifier** model trains an XGBoost classifier (XGBoostClassifier) on the training data (X_train, y_train) with logloss as the evaluation metric.
5. The models were evaluated using:
   - **Accuracy**
   - **Classification Report**
   - **Confusion Matrix**

This gave us a baseline model for performance comparison.

---

### 🔹 Step 8: Model 2 — DistilBERT (Transformer-based Approach)

This step used a deep learning model that understands language context.

1. Tokenization was done using `DistilBertTokenizerFast`.
2. The reviews were converted into:
   - Input IDs
   - Attention masks
3. A pre-trained `DistilBertForSequenceClassification` model was fine-tuned on the dataset.
4. Training used HuggingFace's `Trainer` API and defined training arguments such as:
   - Number of epochs
   - Batch size
   - Learning rate
5. Model performance was then evaluated using the same metrics.

---

### 🔹 Step 9: Evaluation & Results

For both models:
- The performance was measured using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Confusion matrix was visualized using matplotlib and seaborn.
- The BERT-based model provided better understanding of context in reviews and usually showed better performance.

---

## Tools & Libraries Used:
- **Python**
- **Google Colab**
- **Transformers by HuggingFace**
- **Scikit-learn**
- **PyTorch**
- **Plotly, Seaborn, Matplotlib**

---

## How to Run this Project:

1. Open the notebook in **Google Colab**.
2. Mount your **Google Drive** where the dataset is stored.
3. Make sure `drug_review_test.csv` is in the correct path.
4. Run all cells **in order**:
   - Data Loading
   - Visualization
   - Preprocessing
   - Model Training (Logistic Regression and DistilBERT)
   - Evaluation and Output

---

## Summary:
This project demonstrates how textual reviews can be used to predict sentiment or rating using both classical and deep learning techniques. It compares two approaches and visualizes the outcome to draw useful conclusions in drug sentiment analysis.

---

## Author:
Generated from notebook: `drug (4).ipynb`
