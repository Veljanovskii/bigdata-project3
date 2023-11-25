from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()

    spark = SparkSession.builder.appName('PySparkApp - Training').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    HDFS_DATA = os.getenv('HDFS')
    MODEL = os.getenv('MODEL_LOCATION')

    dataFrame = spark.read.csv(HDFS_DATA, header=True)
    dataFrame = dataFrame.withColumn("duration", F.unix_timestamp("ended_at") - F.unix_timestamp("started_at"))
    dataFrame.show()

    columns = ['start_lat', 'start_lng']

    for column in columns:
        dataFrame = dataFrame.withColumn(column, F.col(column).cast(FloatType()))

    vectorAssembler = VectorAssembler().setInputCols(columns).setOutputCol('features').setHandleInvalid('skip')

    assembled = vectorAssembler.transform(dataFrame)

    stringIndexer = StringIndexer().setInputCol('duration').setOutputCol('label')
    indexedDataFrame = stringIndexer.fit(assembled).transform(assembled)

    train_split, test_split = indexedDataFrame.randomSplit([0.8, 0.2])

    print("Starting training")

    regressionModel = LogisticRegression(maxIter=100, regParam=0.02, elasticNetParam=0.8)

    pipeline = Pipeline(stages=[regressionModel])
    regressionModelPipe = pipeline.fit(train_split)

    prediction = regressionModelPipe.transform(test_split)

    evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='prediction',
                                              metricName='areaUnderPR')
    print("Starting evaluation")
    accuracy = evaluator.evaluate(prediction)

    print('Accuracy\'s value for logistic regression model is ' + str(accuracy))

    regressionModelPipe.write().overwrite().save(MODEL)
