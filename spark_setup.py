import os
import findspark
findspark.init()

from pyspark.sql import SparkSession
from config import SPARK_CONFIG, SPARK_EVENT_LOG_DIR, get_logger


logger = get_logger(__name__)

def create_spark_session(app_name=None):
    """Create and return a Spark session"""
    logger.info("Creating Spark session...")

    app_name = app_name or SPARK_CONFIG["app_name"]

    spark_events_dir = os.path.abspath(SPARK_EVENT_LOG_DIR)
    logger.info(f"Using Spark event log directory: {spark_events_dir}")

    session = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", SPARK_CONFIG["driver_memory"])
        .config("spark.executor.memory", SPARK_CONFIG["executor_memory"])
        .config("spark.sql.shuffle.partitions", SPARK_CONFIG["shuffle_partitions"])
        .config("spark.driver.host", SPARK_CONFIG["driver_host"])
        .config("spark.driver.bindAddress", SPARK_CONFIG["bind_address"])
        .config("spark.eventLog.enabled", SPARK_CONFIG["eventLog_enabled"])
        .config("spark.eventLog.dir", SPARK_CONFIG["eventLog_dir"])
        .master(SPARK_CONFIG["master"])
        .getOrCreate()
    )

    session.sparkContext.setLogLevel(SPARK_CONFIG["log_level"])
    
    logger.info(f"Spark session created successfully with app name: {app_name}")
    return session

def word_count_test(spark, input_text="Hello Spark Hello World"):
    """Run a WordCount example to verify Spark setup"""
    logger.info("Running WordCount test...")

    lines_rdd = spark.sparkContext.parallelize([input_text])

    word_counts = (
        lines_rdd
        .flatMap(lambda line: line.split(" "))
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
        .collect()
    )

    logger.info("Word Count Results:")
    for word, count in word_counts:
        logger.info(f"{word}: {count}")
    
    return word_counts

if __name__ == "__main__":
    logger.info("Setting up Spark...")
    spark = create_spark_session()
    
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Spark master: {spark.sparkContext.master}")

    word_count_test(spark)
    
    logger.info("Spark test completed successfully!")
    spark.stop()
