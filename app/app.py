import time
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os
from dotenv import load_dotenv
from influxdb import InfluxDBClient
from pyspark.ml.feature import VectorAssembler, StringIndexer

load_dotenv()

kafka = os.getenv('KAFKA_URL')
topic = os.getenv('KAFKA_TOPIC')
db_name = os.getenv('INFLUXDB_DATABASE')
db_host = os.getenv('INFLUXDB_HOST', 'localhost')
db_port = int(os.getenv('INFLUXDB_PORT'))
db_user = os.getenv('INFLUXDB_USERNAME')
db_password = os.getenv('INFLUXDB_PASSWORD')


def connect_to_influx():
    return InfluxDBClient(db_host, db_port, db_user, db_password, db_name)


def write_to_influx(count, predictions, accuracy):
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    measurement_data = [
        {
            "measurement": topic,
            "time": timestamp,
            "fields": {
                "number of rows": count,
                "predictions": count,
                "correct predictions": accuracy
            }
        }
    ]
    print(measurement_data)
    influxDBConnection.write_points(measurement_data, time_precision='ms')
    print("Count " + str(count))
    print("Accuracy " + str(accuracy))


def analyze(df, epoch, model):
    print("Epoch " + str(epoch))
    columns = ['start_lat', 'start_lng']

    for column in columns:
        df = df.withColumn(column, F.col(column).cast(FloatType()))

    vector_assembler = VectorAssembler().setInputCols(columns).setOutputCol('features').setHandleInvalid('skip')
    assembled = vector_assembler.transform(df)
    string_indexer = StringIndexer().setInputCol('member_casual').setOutputCol('label')
    indexed_data_frame = string_indexer.fit(assembled).transform(assembled)

    prediction = model.transform(indexed_data_frame)

    prediction.select('prediction', 'label')
    # prediction.show(truncate=False)

    predictions_made = prediction.count()
    correct_number = float(prediction.filter(prediction['label'] == prediction['prediction']).count())

    write_to_influx(df.count(), predictions_made, correct_number)


if __name__ == '__main__':
    dataSchema = StructType() \
        .add("ride_id", "string") \
        .add("rideable_type", "string") \
        .add("started_at", "timestamp") \
        .add("ended_at", "timestamp") \
        .add("start_station_name", "string") \
        .add("start_station_id", "string") \
        .add("end_station_name", "string") \
        .add("end_station_id", "string") \
        .add("start_lat", "string") \
        .add("start_lng", "string") \
        .add("end_lat", "string") \
        .add("end_lng", "string") \
        .add("member_casual", "string")

    MODEL = os.getenv('MODEL')
    print("Model Path:", MODEL)
    influxDBConnection = connect_to_influx()

    spark = SparkSession.builder \
        .appName("bigdata-project3-ml-classification") \
        .config("spark.sql.warehouse.dir", MODEL) \
        .config("spark.master", "local[*]") \
        .getOrCreate()

    model = PipelineModel.load(MODEL)
    spark.sparkContext.setLogLevel("INFO")
    sampleDataframe = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .load()
    ).selectExpr("CAST(value as STRING)", "timestamp").select(
        from_json(col("value"), dataSchema).alias("sample"), "timestamp"
    ).select("sample.*")

    sampleDataframe.writeStream \
        .foreachBatch(lambda df, epoch_id: analyze(df, epoch_id, model)) \
        .outputMode("update") \
        .trigger(processingTime="10 seconds") \
        .start().awaitTermination()
