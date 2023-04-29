from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidatorModel


spark = SparkSession.builder\
          .master("local")\
          .appName("CS643_Wine_Quality_Predictions_Project")\
          .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.2")\
          .getOrCreate()


spark.sparkContext._jsc.hadoopConfiguration().set("com.amazonaws.services.s3.enableV4", "true")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider", \
                                     "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.AbstractFileSystem.s3a.impl", "org.apache.hadoop.fs.s3a.S3A")

spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.access.key","ASIAQE4M2SYI6L27GPS7")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.secret.key","3FXYNTgqGueS9lk5nEZl5FmroFSK3X7LGNcaSQVM")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.us-east-1.amazonaws.com")

validation_df = spark.read.format("csv")\
                    .option("header", "true")\
                    .option("inferSchema", "true")\
                    .option("sep", ";")\
                    .load("s3a://mldatawineprediction/ValidationDataset.csv")

if(validation_df.count()>0):
   print("Data loaded successfully")
else:
  print("Something unexpected happend dureing data load")


new_column_names = {
    '"""""fixed acidity""""': 'fixed_acidity',
    '"""fixed acidity""""': 'fixed_acidity',
    '""""volatile acidity""""': 'volatile_acidity',
    '""""citric acid""""': 'citric_acid',
    '""""residual sugar""""': 'residual_sugar',
    '""""chlorides""""': 'chlorides',
    '""""free sulfur dioxide""""': 'free_sulfur_dioxide',
    '""""total sulfur dioxide""""': 'total_sulfur_dioxide',
    '""""density""""': 'density',
    '""""pH""""': 'pH',
    '""""sulphates""""': 'sulphates',
    '""""alcohol""""': 'alcohol',
    '""""quality"""""': 'label'
}

for current_name, new_name in new_column_names.items():
    validation_df = validation_df.withColumnRenamed(current_name, new_name)


model = CrossValidatorModel.load('s3a://mldatawineprediction/LogisticRegression')
evaluator = MulticlassClassificationEvaluator(metricName="f1")
print("F1 Score for LogisticRegression Model: ", evaluator.evaluate(model.transform(validation_df)))


model = CrossValidatorModel.load('s3a://mldatawineprediction/RandomForestClassifier')
evaluator = MulticlassClassificationEvaluator(metricName="f1")
print("F1 Score for RandomForestClassifier Model: ", evaluator.evaluate(model.transform(validation_df)))

model = CrossValidatorModel.load('s3a://mldatawineprediction/DecisionTreeClassifier')
evaluator = MulticlassClassificationEvaluator(metricName="f1")
print("F1 Score for DecisionTreeClassifier Model: ", evaluator.evaluate(model.transform(validation_df)))
