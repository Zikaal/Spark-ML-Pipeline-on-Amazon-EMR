from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Создание Spark сессии
spark = SparkSession.builder.appName("CustomerChurnPipeline").getOrCreate()

print("=" * 80)
print("CUSTOMER CHURN PREDICTION - SPARK ML PIPELINE")
print("=" * 80)

# Загрузка данных
print("\n[1/7] Loading data from HDFS...")
data = spark.read.csv(
    "hdfs:///user/hadoop/churn_input/Churn_Modelling.csv",
    header=True,
    inferSchema=True
)

print(f"Total records: {data.count()}")
print(f"Total columns: {len(data.columns)}")
print("\nSchema:")
data.printSchema()

# Показать примеры данных
print("\nSample data:")
data.show(5)

# Проверка распределения целевой переменной
print("\nChurn distribution:")
data.groupBy("Exited").count().show()

# Split data
print("\n[2/7] Splitting data into train (80%) and test (20%)...")
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
print(f"Training records: {train_data.count()}")
print(f"Test records: {test_data.count()}")

# Categorical encoding
print("\n[3/7] Building pipeline - Categorical encoding...")
geo_indexer = StringIndexer(inputCol="Geography", outputCol="GeographyIndex")
gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")

encoder = OneHotEncoder(
    inputCols=["GeographyIndex", "GenderIndex"],
    outputCols=["GeographyVec", "GenderVec"]
)

# Feature assembly
print("\n[4/7] Building pipeline - Feature assembly...")
assembler = VectorAssembler(
    inputCols=[
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "EstimatedSalary",
        "GeographyVec", "GenderVec"
    ],
    outputCol="features"
)

# Feature scaling
print("\n[5/7] Building pipeline - Feature scaling...")
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures"
)

# Model training
print("\n[6/7] Building pipeline - Model training...")
lr = LogisticRegression(
    labelCol="Exited",
    featuresCol="scaledFeatures",
    maxIter=10
)

# Create pipeline
pipeline = Pipeline(stages=[
    geo_indexer,
    gender_indexer,
    encoder,
    assembler,
    scaler,
    lr
])

# Train model
print("\nTraining Logistic Regression model...")
print("This may take a few minutes on distributed cluster...")
model = pipeline.fit(train_data)
print("✓ Model training completed!")

# Make predictions
print("\n[7/7] Making predictions on test set...")
predictions = model.transform(test_data)

print("\nSample predictions:")
predictions.select("Exited", "prediction", "probability").show(10, truncate=False)

# Evaluation
print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# Accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = accuracy_evaluator.evaluate(predictions)
print(f"\nAccuracy: {accuracy:.4f}")

# Precision
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
precision = precision_evaluator.evaluate(predictions)
print(f"Precision: {precision:.4f}")

# Recall
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="weightedRecall"
)
recall = recall_evaluator.evaluate(predictions)
print(f"Recall: {recall:.4f}")

# F1
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="f1"
)
f1 = f1_evaluator.evaluate(predictions)
print(f"F1 Score: {f1:.4f}")

# AUC
auc_evaluator = BinaryClassificationEvaluator(
    labelCol="Exited",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = auc_evaluator.evaluate(predictions)
print(f"AUC-ROC: {auc:.4f}")

# Confusion matrix
print("\nConfusion Matrix:")
predictions.groupBy("Exited", "prediction").count().show()

print("\n" + "=" * 80)
print("EXPERIMENT: MODEL COMPARISON - Random Forest")
print("=" * 80)

# Random Forest model
print("\nTraining Random Forest model...")
rf = RandomForestClassifier(
    labelCol="Exited",
    featuresCol="scaledFeatures",
    numTrees=10
)

pipeline_rf = Pipeline(stages=[
    geo_indexer,
    gender_indexer,
    encoder,
    assembler,
    scaler,
    rf
])

model_rf = pipeline_rf.fit(train_data)
predictions_rf = model_rf.transform(test_data)
print("✓ Random Forest training completed!")

# Evaluate Random Forest
accuracy_rf = accuracy_evaluator.evaluate(predictions_rf)
f1_rf = f1_evaluator.evaluate(predictions_rf)
auc_rf = auc_evaluator.evaluate(predictions_rf)

print(f"\nRandom Forest Results:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print(f"AUC-ROC: {auc_rf:.4f}")

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print(f"{'Metric':<20} {'Logistic Regression':<25} {'Random Forest':<25}")
print("-" * 80)
print(f"{'Accuracy':<20} {accuracy:<25.4f} {accuracy_rf:<25.4f}")
print(f"{'F1 Score':<20} {f1:<25.4f} {f1_rf:<25.4f}")
print(f"{'AUC-ROC':<20} {auc:<25.4f} {auc_rf:<25.4f}")
print("=" * 80)

# Determine winner
if accuracy_rf > accuracy:
    print("\n✓ Random Forest performs better!")
else:
    print("\n✓ Logistic Regression performs better!")

spark.stop()
print("\nPipeline completed successfully!")
