from pyspark.sql.functions import col, isnan

from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession


spark = SparkSession.builder\
          .master("local")\
          .appName("CS643_Wine_Quality_Predictions_Project")\
          .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.2")\
          .getOrCreate()


spark.sparkContext._jsc.hadoopConfiguration().set("com.amazonaws.services.s3.enableV4", "true")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider", \
                                     "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.AbstractFileSystem.s3a.impl", "org.apache.hadoop.fs.s3a.S3A")

spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.access.key","ASIAQE4M2SYI6L27GPS7")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.secret.key","3FXYNTgqGueS9lk5nEZl5FmroFSK3X7LGNcaSQVM")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.us-east-1.amazonaws.com")


df = spark.read.format("csv")\
      .option("header", "true")\
      .option("inferSchema", "true")\
      .option("sep", ";")\
      .load("s3a://mldatawineprediction/TrainingDataset.csv")

validation_df = spark.read.format("csv")\
                    .option("header", "true")\
                    .option("inferSchema", "true")\
                    .option("sep", ";")\
                    .load("s3a://mldatawineprediction/ValidationDataset.csv")

if(df.count()>0 and validation_df.count()>0):
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
    df = df.withColumnRenamed(current_name, new_name)
    validation_df = validation_df.withColumnRenamed(current_name, new_name)

print(df.columns)
print(validation_df.columns)

null_counts = []
for col_name in df.columns:
    null_count = df.filter(col(col_name).isNull() | isnan(col(col_name))).count()
    null_counts.append((col_name, null_count))

for col_name, null_count in null_counts:
    print(f"Column '{col_name}' has {null_count} null or NaN values.")

df, test_df = df.randomSplit([0.7, 0.3], seed=42)


assembler = VectorAssembler(
    inputCols=['fixed_acidity',
              'volatile_acidity',
              'citric_acid',
              'residual_sugar',
              'chlorides',
              'free_sulfur_dioxide',
              'total_sulfur_dioxide',
              'density',
              'pH',
              'sulphates',
              'alcohol'],
                outputCol="inputFeatures")

# scaler = Normalizer(inputCol="inputFeatures", outputCol="features")
scaler = StandardScaler(inputCol="inputFeatures", outputCol="features")


lr = LogisticRegression()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", seed=0)



pipeline1 = Pipeline(stages=[assembler, scaler, lr])
pipeline2 = Pipeline(stages=[assembler, scaler, rf])
pipeline3 = Pipeline(stages=[assembler, scaler, dt])
paramgrid = ParamGridBuilder().build()
evaluator = MulticlassClassificationEvaluator(metricName="f1")

crossval = CrossValidator(estimator=pipeline1,  
                         estimatorParamMaps=paramgrid,
                         evaluator=evaluator, 
                         numFolds=10
                        )

cvModel1 = crossval.fit(df)
print("F1 Score for LogisticRegression Model: ", evaluator.evaluate(cvModel1.transform(test_df)))

crossval = CrossValidator(estimator=pipeline2,  
                         estimatorParamMaps=paramgrid,
                         evaluator=evaluator, 
                         numFolds=10
                        )

cvModel2 = crossval.fit(df)
print("F1 Score for RandomForestClassifier Model: ", evaluator.evaluate(cvModel2.transform(test_df)))

crossval = CrossValidator(estimator=pipeline3,  
                         estimatorParamMaps=paramgrid,
                         evaluator=evaluator, 
                         numFolds=10
                        )

cvModel3 = crossval.fit(df)
print("F1 Score for DecisionTreeClassifier Model: ", evaluator.evaluate(cvModel3.transform(test_df)))


model_path = "s3a://mldatawineprediction/LogisticRegression"
cvModel1.save(model_path)

model_path = "s3a://mldatawineprediction/RandomForestClassifier"
cvModel2.save(model_path)

model_path = "s3a://mldatawineprediction/DecisionTreeClassifier"
cvModel3.save(model_path)

spark.stop()

