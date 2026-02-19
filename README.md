# Final-Project-Machine-Learning-Model
Machine learning project using the Cleveland Heart Disease dataset to predict the presence of heart disease with Logistic Regression and Random Forest.

# Heart Disease Prediction - Cleveland Dataset

## Problem Definition

Cardiovascular disease is a leading cause of death worldwide.  
This project aims to build a machine learning model that predicts whether a patient has heart disease using clinical and medical features.

This is a binary classification problem:
- 0 = No heart disease
- 1 = Heart disease present

---

## Dataset Description

Dataset: Cleveland Heart Disease Dataset  
Source: UCI Machine Learning Repository  
https://archive.ics.uci.edu/ml/datasets/heart+Disease  

The dataset includes multiple medical features such as:

- Age
- Sex
- Chest pain type
- Cholesterol
- Resting blood pressure
- Maximum heart rate achieved
- ST depression
- Exercise-induced angina

The dataset contains both numerical and categorical (coded) variables, making it suitable for machine learning modeling.

---

## Exploratory Data Analysis (EDA)

EDA included:

- Checking dataset structure and summary statistics
- Verifying missing values
- Analyzing class distribution
- Creating correlation heatmaps

Key Findings:

- The dataset is relatively balanced.
- Certain features such as chest pain type and maximum heart rate show stronger relationships with heart disease.
- Feature relationships suggest non-linear patterns.

---

## Preprocessing

The following preprocessing steps were applied:

1. Separated features and target variable.
2. Applied StandardScaler for feature normalization.
3. Split dataset into training (80%) and testing (20%).
4. Verified absence of missing values.

Feature scaling was necessary because medical measurements vary in magnitude.

---

## Model Implementation

Two models were implemented:

### Logistic Regression
Used as a baseline model due to:
- Binary target variable
- Interpretability
- Simplicity

### Random Forest (Ensemble Method)
Used to:
- Capture non-linear relationships
- Improve generalization
- Reduce overfitting

---

## Model Evaluation

Random Forest achieved:

- Accuracy: 70%
- Recall (Heart Disease): 71%
- Precision: 67%
- F1-score: 69%

The model correctly identifies 71% of heart disease cases. While performance is moderate, results are balanced across both classes.

---

## Strengths

- Balanced classification performance
- Good recall for detecting heart disease
- Ensemble method captures complex relationships

---

## Limitations

- Moderate overall accuracy
- Small dataset size
- Performance may improve with hyperparameter tuning

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Conclusion

This project demonstrates how supervised machine learning can be applied to a real-world medical dataset. The Random Forest model achieved balanced performance and successfully identified most heart disease cases. Further tuning and larger datasets could improve predictive accuracy.
