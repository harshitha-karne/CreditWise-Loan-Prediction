# CreditWise Loan Prediction

An end-to-end supervised machine learning project for predicting loan approval decisions using Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes.

The project includes exploratory data analysis, preprocessing, feature engineering, model benchmarking, and evaluation using business-relevant metrics. Logistic Regression emerged as the best overall model with **88.0% accuracy** and an **F1-score of 0.810**.

---

# Dataset

* 1000 loan applications
* 20 features
* Binary target variable (Loan Approval)
* Mixed numerical and categorical attributes
* Missing values handled using imputation

---

# Model Benchmark

## Baseline Results

| Model                |  Accuracy | Precision |    Recall |  F1-Score |
| -------------------- | --------: | --------: | --------: | --------: |
| Logistic Regression  |     86.5% |     0.783 | **0.770** | **0.777** |
| K-Nearest Neighbors  |     76.0% |     0.627 |     0.525 |     0.571 |
| Gaussian Naive Bayes | **86.5%** | **0.804** |     0.738 |     0.769 |

Gaussian Naive Bayes delivered the strongest baseline precision, while Logistic Regression achieved the best baseline recall and F1-score.

---

# Feature Engineering

EDA indicated that **Credit Score** and **DTI Ratio** contained strong predictive information.

Additional nonlinear features were created:

```python
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
```

## Final Results

| Model                   |  Accuracy | Precision |    Recall |  F1-Score |
| ----------------------- | --------: | --------: | --------: | --------: |
| **Logistic Regression** | **88.0%** |     0.785 | **0.836** | **0.810** |
| K-Nearest Neighbors     |     78.5% |     0.673 |     0.574 |     0.619 |
| Gaussian Naive Bayes    |     86.0% | **0.811** |     0.705 |     0.754 |

---

# Impact of Feature Engineering

| Model                |            Accuracy Change |  F1 Change |
| -------------------- | -------------------------: | ---------: |
| Logistic Regression  | **+1.5 percentage points** | **+0.033** |
| K-Nearest Neighbors  |     +2.5 percentage points |     +0.048 |
| Gaussian Naive Bayes |     -0.5 percentage points |     -0.015 |

Feature engineering improved the performance of Logistic Regression and KNN while demonstrating that transformations do not benefit every algorithm equally.

---

# Model Selection

## Recommended Model: Logistic Regression

Logistic Regression was selected because it provides:

* Highest test accuracy (**88.0%**)
* Highest recall (**83.6%**)
* Highest F1-score (**0.810**)
* Only **10 false negatives** on the test set
* Strong interpretability and simplicity

### Final Confusion Matrix

|            | Predicted No | Predicted Yes |
| ---------- | -----------: | ------------: |
| Actual No  |          125 |            14 |
| Actual Yes |           10 |            51 |

Gaussian Naive Bayes remains an alternative when precision is the primary business objective.

---

# Technical Stack

| Area                    | Technology                                              |
| ----------------------- | ------------------------------------------------------- |
| Programming Language    | Python                                                  |
| Data Manipulation       | Pandas, NumPy                                           |
| Visualization           | Matplotlib, Seaborn                                     |
| Machine Learning        | Scikit-learn                                            |
| Development Environment | Jupyter Notebook                                        |
| Techniques              | Imputation, Encoding, Scaling, EDA, Feature Engineering |

---

# Repository Structure

```text
CreditWise-Loan-Prediction/
│
├── credit_wise_loan.ipynb
├── loan_approval_data.csv
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Run Locally

Clone the repository:

```bash
git clone https://github.com/harshitha-karne/CreditWise-Loan-Prediction.git
cd CreditWise-Loan-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
credit_wise_loan.ipynb
```

and execute the cells sequentially.

---

# What This Project Demonstrates

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Binary classification
* Model benchmarking
* Performance evaluation using precision, recall, and F1-score
* Confusion matrix analysis
* Evidence-based model selection
* Communication of results to stakeholders

---

# Production Roadmap

Future improvements include:

* Scikit-learn Pipelines and ColumnTransformer
* Cross-validation and hyperparameter tuning
* ROC-AUC and Precision-Recall AUC analysis
* Class imbalance handling
* Random Forest, XGBoost, and LightGBM comparison
* SHAP explainability
* Streamlit or FastAPI deployment
* Model monitoring and data drift detection

---

