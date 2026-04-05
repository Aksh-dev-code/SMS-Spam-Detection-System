# SMS Spam Detection System

## 📩 SMS Spam Detection System

## 📌 Overview
This project focuses on building a **Spam Detection Model** using Machine Learning techniques.  
The goal is to classify SMS messages as **Spam** or **Ham (Not Spam)**.

The project includes:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training & Evaluation
- Ensemble Learning (Voting & Stacking)

---

## 🎯 Objective
- Clean and preprocess the dataset
- Perform EDA to understand data patterns
- Extract meaningful features from text
- Train multiple ML models
- Improve performance using ensemble methods
- Identify limitations and improve the system

---

## 📁 Dataset
- **Name:** SMS Spam Collection Dataset  
- **Source:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset  
- Contains labeled SMS messages:
  - `ham` → Not Spam  
  - `spam` → Spam  

---

## 🧹 Data Cleaning
- Removed duplicates
- Dropped unnecessary columns
- Renamed columns for clarity
- Converted labels into numerical format (0 = Ham, 1 = Spam)

---

## 📊 Exploratory Data Analysis (EDA)
- Distribution of Spam vs Ham messages
- Message length analysis:
  - Number of characters
  - Number of words
  - Number of sentences
- Visualization:
  - Histograms
  - Boxplots
  - Pairplots

---

## ⚙️ Feature Engineering
- Text preprocessing:
  - Lowercasing
  - Tokenization
  - Stopword removal
  - Stemming/Lemmatization
- Vectorization:
  - TF-IDF (Term Frequency–Inverse Document Frequency)

---

## 🤖 Models Used
The following machine learning models were implemented:

- Bernoulli Naive Bayes (BNB)
- Multinomial Naive Bayes (MNB)
- Gaussian Naive Bayes (GNB)
- Support Vector Machine (SVM)
- Random Forest Classifier (RFC)
- Logistic Regression
- Gradient Boosting Classifier (GBC)
- Bagging Classifier
- AdaBoost Classifier
- Decision Tree Classifier (DTC)

---

## 🔗 Ensemble Learning
To improve performance, the following techniques were applied:

- **Voting Classifier**
- **Stacking Classifier**

---

## 📈 Results & Observations
- The model performs well on **Ham messages**
- However, some **Spam messages are misclassified as Ham**
- This indicates:
  - Class imbalance issue
  - Feature limitations
  - Need for better text representation

- The model sometimes gives **errors when tested on new/unseen data**, likely due to:
  - Inconsistent preprocessing
  - Vocabulary mismatch in TF-IDF
  - Overfitting

---

## ⚠️ Limitations
- Basic feature engineering (TF-IDF only)
- No deep contextual understanding of text
- Model struggles with unseen or complex messages
- Slight overfitting in ensemble models

---

## 🚀 Future Improvements
- Improve feature engineering
- Handle class imbalance better
- Use advanced NLP techniques:
  - Word embeddings (Word2Vec, GloVe)
- Apply Transformer-based models:
  - **BERT (Bidirectional Encoder Representations from Transformers)**
- Improve preprocessing pipeline consistency
- Deploy model with better generalization

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- NLTK

---

## 📌 Conclusion
This project demonstrates a complete ML pipeline for spam detection.  
While traditional ML models perform reasonably well, **modern NLP models like BERT can significantly improve accuracy and generalization**.

---

## Author
Aanuska Maity