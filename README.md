# Bank Customer Churn Prediction - Spark ML Pipeline

## Dataset
Bank Customer Churn Dataset from Kaggle
- 10,000 customers
- Target: Exited (0=No churn, 1=Churn)

## Features
**Categorical:**
- Geography (France, Germany, Spain)
- Gender (Male, Female)

**Numerical:**
- CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary

## Pipeline Stages
1. StringIndexer (Geography, Gender)
2. OneHotEncoder
3. VectorAssembler
4. StandardScaler
5. LogisticRegression / RandomForestClassifier

## Running on EMR

### Upload dataset to HDFS:
```bash
hdfs dfs -put Churn_Modelling.csv /user/hadoop/churn_input/
```

### Submit Spark job:
```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  churn_pipeline.py
```

## Experiment
Model Comparison: Logistic Regression vs Random Forest
- Both models trained on same features
- Evaluated using Accuracy, F1, AUC-ROC
