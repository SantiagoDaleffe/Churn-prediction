# **Customer Churn & Risk Classification Model**

## **Project Overview**

This project implements a high-precision classification system designed to identify high-risk user behavior and predict customer churn. The methodology applied here-focusing on behavioral patterns and anomaly detection-is directly transferable to Fraud Detection domains.

The model achieves a robust **0.75+ AUC score** on unseen data, utilizing advanced Feature Engineering and Gradient Boosting techniques to handle complex, imbalanced transactional data.

## **Key Features & Techniques**

- **Behavioral Feature Engineering:** Transformed raw transactional data into behavioral ratios (e.g., CostPerHour, ComplaintIntensity) to detect subtle patterns indicative of attrition or anomalous activity.
- **Imbalanced Dataset Handling:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) and class-weight adjustments within a robust pipeline to improve recall on the minority class (Churners).
- **Robust Validation:** Used **Stratified K-Fold Cross-Validation** to ensure performance stability and prevent data leakage.
- **Advanced Modeling:** Leveraged **XGBoost** and **CatBoost** with custom threshold tuning to maximize the F1-Score and Recall, prioritizing the identification of high-risk users.
- **Target Encoding:** Applied M-Estimate Target Encoding to capture signals from high-cardinality categorical features without overfitting.

## **Repository Structure**
```bash
├── 01_Feature_Engineering_and_Preprocessing.ipynb # Data cleaning, behavioral FE, and Target Encoding

├── 02_Model_Training_and_Evaluation.ipynb # Model training (XGBoost), CV, and Threshold Tuning

├── data/

│ ├── train.csv # Raw training data

│ ├── test.csv # Raw testing data

│ ├── data_descriptions.csv # Feature dictionary

│ ├── train_cleaned.csv # Processed training set

│ └── test_cleaned.csv # Processed testing set

│ └── prediction_submission.csv # Final probability predictions on test set

└── README.md
```

## **How to Run**

- **Install Dependencies:** Run the following command to install the required libraries:
```bash
- pip install pandas numpy scikit-learn xgboost category_encoders imbalanced-learn
```
- **Preprocessing:** Run 01_Feature_Engineering_and_Preprocessing.ipynb to generate the engineered datasets.
- **Modeling:** Run 02_Model_Training_and_Evaluation.ipynb to train the model, visualize performance (ROC-AUC, Confusion Matrix), and generate predictions.

## **Tech Stack**

- **Python**
- **Scikit-Learn** (Pipelines, Validation)
- **XGBoost / CatBoost** (Gradient Boosting Classifiers)
- **Imbalanced-Learn** (SMOTE)
- **Category Encoders** (Target Encoding)
- **Pandas & NumPy** (Data Manipulation)

### _Author: Santiago Daleffe_