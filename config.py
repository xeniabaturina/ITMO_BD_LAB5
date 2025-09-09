import os
import logging
from logging.handlers import RotatingFileHandler

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Data subdirectories
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
SPARK_EVENT_LOG_DIR = os.path.join(DATA_DIR, "spark-events")

# Create all directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                  REPORTS_DIR, PLOTS_DIR, LOGS_DIR, SPARK_EVENT_LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data file paths
ORIGINAL_DATA_PATH = os.path.join(RAW_DATA_DIR, "openfoodfacts-mongodbdump.tar.gz")
MONGO_EXPORT_PATH = os.path.join(RAW_DATA_DIR, "products.json")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_food_data.parquet")
CSV_DATA_PATH = os.path.join(RAW_DATA_DIR, "openfoodfacts-products.csv")

# Model paths
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model")

# Report paths
REPORT_PATH = os.path.join(REPORTS_DIR, "clustering_report.txt")

# Spark configuration
SPARK_CONFIG = {
    "app_name": "FoodClusteringApp",
    "master": "local[*]",  # Use all available cores
    "driver_memory": "4g",  # Driver memory allocation
    "executor_memory": "4g",  # Executor memory allocation
    "shuffle_partitions": "20",  # Number of partitions for shuffling
    "driver_host": "localhost",  # Driver host
    "bind_address": "127.0.0.1",  # Bind address
    "log_level": "ERROR",  # Spark log level
    "eventLog_enabled": "true",  # Enable event logging
    "eventLog_dir": SPARK_EVENT_LOG_DIR,  # Event log directory
}

# K-means configuration
KMEANS_CONFIG = {
    "k": 5,  # Number of clusters
    "seed": 42,  # Random seed for reproducibility
    "max_iter": 200,  # Maximum iterations for convergence
    "tol": 1e-6,  # Convergence tolerance
    "init_steps": 10  # Number of steps for initialization
}

# Feature configuration
FEATURE_CONFIG = {
    "feature_cols": [
        "energy_100g", "proteins_100g", "carbohydrates_100g", 
        "sugars_100g", "fat_100g", "saturated-fat_100g",
        "fiber_100g", "salt_100g", "sodium_100g"
    ],
    "outlier_percentile": [0.02, 0.98],  # Percentiles for outlier removal
    "outlier_approx_error": 0.01  # Approximate error for percentile calculation
}

# Data processing configuration
DATA_CONFIG = {
    "urls": [
        "https://static.openfoodfacts.org/data/openfoodfacts-mongodbdump.tar.gz",
        "https://world.openfoodfacts.org/data/en.openfoodfacts.org.products.csv"
    ],
    "chunk_size": 1024 * 1024,  # 1MB chunks for downloading
    "min_valid_fields": 5,  # Minimum number of valid nutritional fields required
    "max_records": 200000  # Maximum number of records to process
}

# Logging configuration
LOG_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file": os.path.join(LOGS_DIR, "food_clustering.log"),
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Custom formatter for aligned log output
class AlignedFormatter(logging.Formatter):
    def format(self, record):
        # Pad name and levelname to fixed width
        record.name = f"{record.name:<18}"  # 18 chars for module
        record.levelname = f"{record.levelname:<8}"  # 8 chars for level
        return super().format(record)

# Set up logging
def setup_logging(name=None):
    """Set up logging with the specified configuration."""
    logger = logging.getLogger(name or __name__)
    logger.setLevel(LOG_CONFIG["level"])
    
    # Create aligned formatter
    formatter = AlignedFormatter(
        fmt=LOG_CONFIG["format"],
        datefmt=LOG_CONFIG["date_format"]
    )
    
    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        LOG_CONFIG["file"],
        maxBytes=LOG_CONFIG["max_bytes"],
        backupCount=LOG_CONFIG["backup_count"]
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def clear_logs():
    """Clear the current log file to start fresh for a new run."""
    log_file = LOG_CONFIG["file"]
    if os.path.exists(log_file):
        try:
            # Clear the file content
            with open(log_file, 'w') as f:
                f.write("")
            print(f"Cleared log file: {log_file}")
        except Exception as e:
            print(f"Warning: Could not clear log file {log_file}: {e}")

def log_dataframe_show(logger, df, description="DataFrame", num_rows=20):
    """
    Capture Spark DataFrame show() output and log it.
    
    Args:
        logger: Logger instance to write to
        df: Spark DataFrame
        description: Description of the DataFrame
        num_rows: Number of rows to show
    """
    try:
        # Convert DataFrame show to string by collecting output
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Capture the show() output
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            df.show(num_rows, truncate=False)
        
        # Log the captured output
        output_str = output_buffer.getvalue()
        logger.info(f"{description}:")
        for line in output_str.strip().split('\n'):
            if line.strip():  # Only log non-empty lines
                logger.info(line)
                
    except Exception as e:
        logger.warning(f"Could not capture DataFrame output for {description}: {e}")
        # Fallback to regular show
        df.show(num_rows)

# Default logger
logger = setup_logging()

def get_logger(name):
    """Get a logger with the specified name."""
    return setup_logging(name)
