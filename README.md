# Banking Customer Churn Prediction

A comprehensive machine learning project for predicting customer churn in banking. This project implements end-to-end data analysis, preprocessing, feature engineering, and model development to identify customers at risk of leaving the bank.

## 📋 Project Overview

**Goal**: Build a predictive model to identify bank customers who are likely to exit/churn, enabling proactive retention strategies.

**Dataset**: Bank Customer Churn dataset from [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

**Techniques**: Exploratory Data Analysis, Feature Engineering, Data Preprocessing, Machine Learning Classification

## 📊 Dataset Description

The dataset contains **10,000 customer records** with the following features:

### Features:
- **Customer Information**: CustomerId, Surname, Geography (France, Spain, Germany), Gender
- **Financial**: CreditScore, EstimatedSalary, Balance, NumOfProducts
- **Banking Activity**: Tenure (years), IsActiveMember, HasCrCard
- **Card Information**: Card Type (DIAMOND, GOLD, SILVER, PLATINUM)
- **Customer Satisfaction**: Satisfaction Score, Point Earned, Complain

### Target Variable:
- **Exited**: Binary indicator (1 = churned, 0 = retained)

## 🗂️ Project Structure

```
banking_churn_pred/
├── 1. EDA.ipynb                    # Exploratory Data Analysis
├── 2. data_preproces.ipynb        # Data Preprocessing & Feature Engineering
├── 3. modeling.ipynb              # Model Development & Hyperparameter Tuning
├── 4. final_results.ipynb         # Results Summary (Empty - to be filled)
├── utils.py                       # Utility functions for preprocessing
├── data.csv                       # Raw dataset
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔍 Notebook Breakdown

### 1. **Exploratory Data Analysis (EDA)** - `1. EDA.ipynb`

Comprehensive analysis of the dataset structure and relationships:

- **Data Inspection**: Shape, columns, data types, missing values
- **Statistical Summary**: Descriptive statistics for all features
- **Target Distribution**: Class balance analysis of the Exited variable
- **Univariate Analysis**: Distribution plots for numerical features
- **Feature vs Target Analysis**: 
  - Box plots showing numerical features vs churn
  - Count plots for categorical features vs churn
- **Correlation Analysis**: 
  - Correlation matrix heatmap
  - Identification of features most correlated with churn
  - Detection of highly correlated features (e.g., Complain ≈ 100% correlated with Exited)

**Key Finding**: The "Complain" variable is almost perfectly correlated with "Exited" and must be excluded to ensure model generalizability.

---

### 2. **Data Preprocessing** - `2. data_preproces.ipynb`

Data preparation and feature engineering pipeline:

#### Data Cleaning:
- Drops unnecessary columns: RowNumber, CustomerId, Surname
- Removes "Complain" due to target leakage
- Separates features (X) and target (y)

#### Feature Engineering - Three Derived Variables:

1. **BalanceSalaryRatio** = Balance / EstimatedSalary

2. **TenureByAge** = Tenure / Age

3. **CreditScoreGivenAge** = CreditScore / Age

#### Preprocessing Pipeline (Scikit-Learn):

Uses `ColumnTransformer` and `Pipeline` for automated preprocessing:

- **Numerical Features**: Passed through unchanged
- **Multi-Class Categorical**: One-Hot Encoded (e.g., Geography, Card Type)
- **Binary Categorical**: Ordinal Encoded (0/1)
- **Scaling**: StandardScaler (mean=0, std=1) applied to entire dataset

---

### 3. **Model Development & Tuning** - `3. modeling.ipynb`

Three classification algorithms evaluated with hyperparameter optimization:

#### Data Split:
- 80% Training / 20% Testing
- **Stratified split** (stratify=y) maintains class proportions in both sets

#### Models Tested:

- **Support Vector Machine (SVM)**

- **Decision Tree Classifier**


- **Random Forest Classifier**


#### Optimization Strategy:

- **Grid Search with 5-Fold Cross-Validation**
- **Optimization Metric**: F1-Score (better than Accuracy for imbalanced data)
- **Class Weight**: 'balanced' to handle class imbalance

#### Evaluation Metrics:

For each model, computed on test set:
- **Accuracy**: Overall correctness
- **Precision**: Reliability of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision & recall
- **Confusion Matrix**: Visualization of prediction breakdown

---

### 4. **Final Results** - `4. final_results.ipynb`

*To be populated with*: Model comparison summary, best model selection, feature importance analysis, business recommendations.

## 🛠️ Technologies & Dependencies

### Core Libraries:
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and preprocessing
- **matplotlib** - Basic plotting
- **seaborn** - Statistical data visualization

See [requirements.txt](requirements.txt) for complete dependency list with versions.

## 🚀 Getting Started

### Installation:

```bash
# Clone the repository
git clone <https://github.com/chillMLguy/banking_churn_pred?tab=readme-ov-file>

# Create virtual environment 
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks:

```bash
# Start Jupyter
jupyter notebook

# Execute notebooks in order:
# 1. EDA.ipynb
# 2. data_preproces.ipynb
# utils.py
# 3. modeling.ipynb
# 4. final_results.ipynb
```

## 🎯 Key Insights & Findings

1. **Complain Variable**: Dropped due to perfect target leakage (~100% correlation with Exited)
2. **Feature Engineering**: Three derived features capture behavioral patterns not visible in raw data
3. **Class Imbalance**: Addressed through stratified splitting and balanced class weights
4. **Model Evaluation**: F1-Score prioritized over Accuracy for meaningful metric
5. **Best Model**

## 📝 Project Information

**Context**: Machine Learning for Finance class project

## 📧 Contact & Attribution

Dataset source: [Bank Customer Churn - Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
