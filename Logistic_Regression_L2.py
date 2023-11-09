from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import sys

spark = SparkSession.builder \
    .appName("Australian Court Case Classification") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

train_data = spark.read.text(sys.argv[1]).withColumnRenamed("value", "text")

tokenizer = Tokenizer(inputCol="text", outputCol="words")
train_words_data = tokenizer.transform(train_data)

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
train_filtered_data = remover.transform(train_words_data)

count_vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features", vocabSize=20000)
idf = IDF(inputCol="raw_features", outputCol="features")

pipeline = Pipeline(stages=[count_vectorizer, idf])
model = pipeline.fit(train_filtered_data)
train_tfidf_data = model.transform(train_filtered_data)

vocabulary = model.stages[0].vocabulary

word_frequencies = dict(zip(vocabulary, range(len(vocabulary))))

def calculate_average_tfidf(documents, words):
  total_tfidf = [0] * len(words)
  num_documents = documents.count()

  for word in words:
    word_index = word_frequencies.get(word, -1)
    if 0 <= word_index < len(total_tfidf):
      word_tfidf = documents.rdd.map(lambda x: x.features[word_index]).sum()
      total_tfidf[word_index] = word_tfidf / num_documents

  return total_tfidf

court_documents = train_tfidf_data.filter(train_tfidf_data.text.contains("<doc id=\"AU"))
court_tfidf_values = calculate_average_tfidf(court_documents, ["applicant", "and", "attack", "protein", "court"])

wiki_documents = train_tfidf_data.filter(~train_tfidf_data.text.contains("<doc id=\"AU"))
wiki_tfidf_values = calculate_average_tfidf(wiki_documents, ["applicant", "and", "attack", "protein", "court"])

print("Average TF-IDF values for court documents:", court_tfidf_values)
print("Average TF-IDF values for Wikipedia documents:", wiki_tfidf_values)

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train_logistic_regression(train_data, num_iterations=100, reg_param=0.1):
  lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=num_iterations, regParam=reg_param, elasticNetParam=0.0)
  lr_model = lr.fit(train_data)
  return lr_model

from pyspark.sql import functions as F

court_labeled_data = court_documents.withColumn('label', F.lit(1))
wiki_labeled_data = wiki_documents.withColumn('label', F.lit(0))
training_data = court_labeled_data.union(wiki_labeled_data).cache()

lr_model = train_logistic_regression(training_data)

coefficients = lr_model.coefficients.toArray()
vocabulary_with_coefficients = list(zip(vocabulary, coefficients))
top_5_words = sorted(vocabulary_with_coefficients, key=lambda x: abs(x[1]), reverse=True)[:5]
print("Top 5 words with the largest regression coefficients:")

for word, coefficient in top_5_words:
  print(f"{word}: {coefficient}")

test_data = spark.read.text(sys.argv[2]).withColumnRenamed("value", "text")

test_words_data = tokenizer.transform(test_data)

test_words_data = remover.transform(test_words_data)

test_tfidf_data = model.transform(test_words_data)

test_labeled_data = test_tfidf_data.withColumn('label', F.when(test_tfidf_data.text.contains("<doc id=\"AU"), F.lit(1)).otherwise(F.lit(0)))

test_labeled_data.cache()

test_predictions = lr_model.transform(test_labeled_data)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol='label')

auc = evaluator.evaluate(test_predictions)

print("Area under ROC:", auc)

# Get false positives (predicted as Australian court case but actually Wikipedia article)
false_positives = test_predictions.filter((F.col("prediction") == 1) & (F.col("label") == 0))

# Limit the number of false positives to 3
false_positives = false_positives.limit(3)

print("False Positives (Wikipedia articles wrongly predicted as Australian court cases):")

for row in false_positives.select("text").collect():
  print(row.text)

predictionAndLabels = test_predictions.select("prediction", "label").rdd.map(lambda row: (float(row.prediction), float(row.label)))

from pyspark.mllib.evaluation import MulticlassMetrics

# Instantiate MulticlassMetrics
metrics = MulticlassMetrics(predictionAndLabels)

f1_score = metrics.fMeasure(1.0)
precision = metrics.precision(1.0)
recall = metrics.recall(1.0)
accuracy = metrics.accuracy

print("F1 Score:", f1_score)
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)

spark.stop()