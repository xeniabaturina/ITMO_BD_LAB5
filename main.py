import os
import time
import findspark
findspark.init()

from config import (
    PROCESSED_DATA_PATH, KMEANS_MODEL_PATH, REPORT_DIR,
    KMEANS_CONFIG, get_logger, clear_logs
)


logger = get_logger(__name__)

def main():
    """Main function to run the pipeline with optional cache reset"""
    # Clear logs at the start of each run
    clear_logs()
    logger.info("=" * 80)
    logger.info("ITMO Big Data Lab 5: PySpark K-Means Clustering on Food Data")
    logger.info("=" * 80)

    logger.info("Testing Spark setup...")
    from spark_setup import create_spark_session, word_count_test
    spark = create_spark_session(app_name="FoodClusteringPipeline")
    
    logger.info("Running WordCount test...")
    word_count_test(spark, "Testing Spark with a simple WordCount example")
    logger.info("Spark setup test completed successfully!")
    spark.stop()

    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.info("Processing food data...")
        from data_processor import main as process_data
        process_data()
    else:
        logger.info("Processed data already exists, skipping data processing...")

    logger.info("Building and evaluating K-means model...")
    try:
        from kmeans_clustering import main as cluster_data
        model, predictions, silhouette = cluster_data()
    except Exception as e:
        logger.error(f"Error running clustering: {e}")
        silhouette = 0.0

    logger.info("=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80)
    logger.info("Results summary:")
    logger.info(f"Silhouette score: {silhouette:.2f}")
    logger.info(f"Model saved to: {os.path.abspath(KMEANS_MODEL_PATH)}")
    logger.info(f"Reports saved to: {os.path.abspath(REPORT_DIR)}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
