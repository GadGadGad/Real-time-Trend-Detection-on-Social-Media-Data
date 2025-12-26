from pyspark.sql import SparkSession
import os

def get_spark_session(app_name="TrendAnalysis"):
    # Tự động tải các package Kafka và Postgres khi khởi chạy
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.postgresql:postgresql:42.6.0 pyspark-shell'
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    return spark