import os
import tempfile
import requests
import tarfile
import findspark
findspark.init()
from tqdm import tqdm

from pyspark.sql.functions import col, when, isnan, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from config import (
    DATA_DIR, ORIGINAL_DATA_PATH, MONGO_EXPORT_PATH, 
    PROCESSED_DATA_PATH, CSV_DATA_PATH,
    get_logger, log_dataframe_show
)
from spark_setup import create_spark_session

logger = get_logger(__name__)


def download_dataset(url="https://static.openfoodfacts.org/data/openfoodfacts-mongodbdump.tar.gz"):
    """Download the Open Food Facts dataset with a tqdm progress bar."""
    global MONGO_EXPORT_PATH

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"Processed data already exists at {PROCESSED_DATA_PATH}")
        return PROCESSED_DATA_PATH

    if os.path.exists(ORIGINAL_DATA_PATH) and os.path.getsize(ORIGINAL_DATA_PATH) > 1000000:
        print(f"Dataset already downloaded at {ORIGINAL_DATA_PATH}")
        return ORIGINAL_DATA_PATH

    if os.path.exists(CSV_DATA_PATH) and os.path.getsize(CSV_DATA_PATH) > 1000000:
        print(f"CSV dataset already downloaded at {CSV_DATA_PATH}")
        MONGO_EXPORT_PATH = CSV_DATA_PATH
        return CSV_DATA_PATH
    
    urls = [
        "https://static.openfoodfacts.org/data/openfoodfacts-mongodbdump.tar.gz",
        "https://world.openfoodfacts.org/data/en.openfoodfacts.org.products.csv",
    ]
    
    if url not in urls:
        urls.insert(0, url)
    
    for current_url in urls:
        try:
            print(f"Attempting to download dataset from {current_url}...")

            session = requests.Session()

            head_response = session.head(current_url, allow_redirects=True)
            if head_response.status_code != 200:
                print(f"Failed to access {current_url}, status code: {head_response.status_code}")
                continue

            content_type = head_response.headers.get('Content-Type', '')
            if 'html' in content_type.lower():
                print(f"Warning: URL returns HTML content instead of a data file. Content-Type: {content_type}")

            output_path = ORIGINAL_DATA_PATH
            if current_url.endswith('.csv'):
                output_path = CSV_DATA_PATH

            response = session.get(current_url, stream=True)
            if response.status_code != 200:
                print(f"Failed to download from {current_url}, status code: {response.status_code}")
                continue
            
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024 * 1024
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                
                desc = f"Downloading {os.path.basename(current_url)}"
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc, ncols=100) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            temp_file.write(chunk)
                            pbar.update(len(chunk))
            
            is_valid = True
            
            if os.path.getsize(temp_path) < 10 * 1024 * 1024:
                file_extension = current_url.split('.')[-1].lower()
                
                if file_extension == 'gz':
                    try:
                        with open(temp_path, 'rb') as f:
                            magic_number = f.read(2)
                            is_valid = magic_number == b'\x1f\x8b'
                    except Exception as e:
                        print(f"Error validating gzip file: {e}")
                        is_valid = False
            
            if is_valid:
                import shutil
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.move(temp_path, output_path)
                print(f"Download completed and saved to: {output_path}")
                
                if current_url.endswith('.csv'):
                    MONGO_EXPORT_PATH = output_path
                    return output_path
                    
                return output_path
            else:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                print(f"Downloaded file from {current_url} appears to be invalid.")
        except Exception as e:
            print(f"Error downloading from {current_url}: {e}")
    
    raise RuntimeError("Failed to download dataset from any of the provided URLs.")

def extract_dataset(tar_path):
    """Extract the dataset or prepare it for processing depending on file type."""
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Dataset file not found: {tar_path}")
    
    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"Processed data already exists at {PROCESSED_DATA_PATH}")
        return PROCESSED_DATA_PATH
    
    if os.path.exists(MONGO_EXPORT_PATH):
        print(f"Dataset already extracted at {MONGO_EXPORT_PATH}")
        return MONGO_EXPORT_PATH
    
    if tar_path.endswith('.csv'):
        print(f"File is already in CSV format, no extraction needed: {tar_path}")
        return tar_path
    
    print(f"Extracting dataset from {tar_path}...")
    try:
        with open(tar_path, 'rb') as f:
            magic_number = f.read(2)
            if magic_number != b'\x1f\x8b':
                print("Warning: File does not appear to be a valid gzip file.")
                print("Attempting to download a different dataset format...")
                return download_alternative_data()
        
        with tarfile.open(tar_path, "r:gz") as tar:
            json_found = False
            for member in tar.getmembers():
                if member.name.endswith("products.json"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=DATA_DIR)
                    extracted_path = os.path.join(DATA_DIR, member.name)
                    print(f"Extracted {member.name} to {extracted_path}")
                    json_found = True
                    return extracted_path
            
            if not json_found:
                print("Could not find products.json in the archive.")
                return download_alternative_data()
                
    except Exception as e:
        print(f"Error extracting tar.gz file: {e}")
        return download_alternative_data()
        
def download_alternative_data():
    """Download alternative data format (CSV) with tqdm progress bar."""
    global MONGO_EXPORT_PATH
    
    print("Downloading alternative dataset in CSV format...")
    
    csv_url = "https://world.openfoodfacts.org/data/en.openfoodfacts.org.products.csv"
    
    if os.path.exists(CSV_DATA_PATH) and os.path.getsize(CSV_DATA_PATH) > 1000000:
        print(f"Alternative dataset already exists at {CSV_DATA_PATH}")
        MONGO_EXPORT_PATH = CSV_DATA_PATH
        return CSV_DATA_PATH
        
    try:
        print(f"Downloading CSV file from {csv_url}...")
        session = requests.Session()
        response = session.get(csv_url, stream=True)
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download CSV, status code: {response.status_code}")
            
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024 * 1024
        
        desc = "Downloading CSV dataset"
        with open(CSV_DATA_PATH, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc, ncols=100) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"CSV file downloaded to {CSV_DATA_PATH}")
        
        MONGO_EXPORT_PATH = CSV_DATA_PATH
        
        return CSV_DATA_PATH
        
    except Exception as e:
        print(f"Error downloading alternative dataset: {e}")
        raise RuntimeError("Failed to download and extract any compatible dataset format.")

def preprocess_data(spark, data_path):
    """Load and preprocess the dataset using PySpark."""
    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"Processed data already exists at {PROCESSED_DATA_PATH}")
        return spark.read.parquet(PROCESSED_DATA_PATH)
    
    print(f"Loading dataset from {data_path}...")
    
    file_format = data_path.split('.')[-1].lower()
    
    if file_format == 'json':
        schema = StructType([
            StructField("_id", StringType(), True),
            StructField("product_name", StringType(), True),
            StructField("brands", StringType(), True),
            StructField("categories", StringType(), True),
            StructField("nutriments", StructType([
                StructField("energy_100g", DoubleType(), True),
                StructField("proteins_100g", DoubleType(), True),
                StructField("carbohydrates_100g", DoubleType(), True),
                StructField("sugars_100g", DoubleType(), True),
                StructField("fat_100g", DoubleType(), True),
                StructField("saturated-fat_100g", DoubleType(), True),
                StructField("fiber_100g", DoubleType(), True),
                StructField("salt_100g", DoubleType(), True),
                StructField("sodium_100g", DoubleType(), True),
            ]), True),
            StructField("nutriscore_grade", StringType(), True),
            StructField("nova_group", StringType(), True),
        ])
        
        try:
            with open(data_path, 'r') as file:
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                
                sample_lines = 100
                sample_size = 0
                for i in range(sample_lines):
                    line = file.readline()
                    if not line:
                        break
                    sample_size += len(line)
                
                if sample_size > 0:
                    estimated_lines = int((file_size / sample_size) * sample_lines)
                else:
                    estimated_lines = 1000000
                
                file.seek(0)
                
                max_lines = 200000
                
                lines = []
                with tqdm(total=min(max_lines, estimated_lines), desc="Reading JSON data", unit="lines", ncols=100) as pbar:
                    for i, line in enumerate(file):
                        if i >= max_lines:
                            break
                        lines.append(line)
                        pbar.update(1)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
                temp_path = temp_file.name
                temp_file.write('\n'.join(lines).encode('utf-8'))
            
            print(f"Loading {len(lines)} products into DataFrame...")
            df = spark.read.json(temp_path)
            
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            df = spark.read.format("json").load(data_path).limit(100000)
        
        print(f"Loaded {df.count()} products. Preprocessing data...")
        
        nutrient_cols = [
            "nutriments.energy_100g", 
            "nutriments.proteins_100g", 
            "nutriments.carbohydrates_100g",
            "nutriments.sugars_100g", 
            "nutriments.fat_100g", 
            "nutriments.saturated-fat_100g",
            "nutriments.fiber_100g", 
            "nutriments.salt_100g", 
            "nutriments.sodium_100g"
        ]
        
        flattened_df = df.select(
            "_id",
            "product_name",
            "brands",
            "categories",
            "nutriscore_grade",
            "nova_group",
            *nutrient_cols
        )
        
        for col_name in nutrient_cols:
            simple_name = col_name.split('.')[-1]
            flattened_df = flattened_df.withColumnRenamed(col_name, simple_name)
        
    elif file_format == 'csv':
        print("Loading CSV data...")
        try:
            df = spark.read.format("csv") \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .option("delimiter", "\t") \
                .load(data_path) \
                .limit(200000)
                
            print(f"Loaded {df.count()} products from CSV. Preprocessing data...")
            
            column_mapping = {
                "code": "_id",
                "product_name": "product_name",
                "brands": "brands",
                "categories": "categories",
                "nutriscore_grade": "nutriscore_grade",
                "nova_group": "nova_group",
                "energy_100g": "energy_100g",
                "proteins_100g": "proteins_100g",
                "carbohydrates_100g": "carbohydrates_100g",
                "sugars_100g": "sugars_100g",
                "fat_100g": "fat_100g",
                "saturated-fat_100g": "saturated-fat_100g",
                "fiber_100g": "fiber_100g",
                "salt_100g": "salt_100g",
                "sodium_100g": "sodium_100g"
            }
            
            flattened_df = df
            for original, new_name in column_mapping.items():
                if original in df.columns:
                    flattened_df = flattened_df.withColumnRenamed(original, new_name)
            
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            print("Trying with comma delimiter...")
            df = spark.read.format("csv") \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .option("delimiter", ",") \
                .load(data_path) \
                .limit(200000)
                
            print(f"Loaded {df.count()} products from CSV. Preprocessing data...")
            flattened_df = df
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    nutrient_columns = [
        "energy_100g", "proteins_100g", "carbohydrates_100g", 
        "sugars_100g", "fat_100g", "saturated-fat_100g",
        "fiber_100g", "salt_100g", "sodium_100g"
    ]
    
    existing_nutrient_columns = [col for col in nutrient_columns if col in flattened_df.columns]
    
    if len(existing_nutrient_columns) < 5:
        print(f"Warning: Only {len(existing_nutrient_columns)} out of 9 nutrient columns found in the data.")
        print("Adding missing nutrient columns with default values of 0.")
        
        for col_name in nutrient_columns:
            if col_name not in flattened_df.columns:
                flattened_df = flattened_df.withColumn(col_name, col(col_name).cast(DoubleType()))
    
    for col_name in nutrient_columns:
        flattened_df = flattened_df.withColumn(
            f"{col_name}_valid", 
            when(col(col_name).isNotNull() & (~isnan(col(col_name))), 1).otherwise(0)
        )
    
    valid_cols = [f"{col}_valid" for col in nutrient_columns]
    flattened_df = flattened_df.withColumn(
        "valid_nutrient_count", 
        sum([col(c) for c in valid_cols])
    )
    
    filtered_df = flattened_df.filter(col("valid_nutrient_count") >= 5)
    filtered_df = filtered_df.drop(*valid_cols, "valid_nutrient_count")
    
    for col_name in nutrient_columns:
        filtered_df = filtered_df.withColumn(
            col_name, 
            when(col(col_name).isNull() | isnan(col(col_name)), 0).otherwise(col(col_name))
        )
    
    required_columns = ["_id", "product_name", "brands", "categories", "nutriscore_grade", "nova_group"] + nutrient_columns
    for col_name in required_columns:
        if col_name not in filtered_df.columns:
            filtered_df = filtered_df.withColumn(col_name, lit(None))
    
    final_df = filtered_df.select(*required_columns)
    
    print(f"Preprocessing complete. Final dataset has {final_df.count()} products.")
    
    print(f"Saving processed data to {PROCESSED_DATA_PATH}...")
    final_df.write.mode("overwrite").parquet(PROCESSED_DATA_PATH)
    
    return final_df

def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        if os.path.exists(PROCESSED_DATA_PATH):
            print(f"Processed data already exists at {PROCESSED_DATA_PATH}")
            df = spark.read.parquet(PROCESSED_DATA_PATH)
            print("Loaded existing processed data.")
            return df
        tar_path = download_dataset()
        json_path = extract_dataset(tar_path)
        df = preprocess_data(spark, json_path)
        print("\nSample of processed data:")
        log_dataframe_show(logger, df, "Sample of processed data", num_rows=5)
        print("\nSummary statistics:")
        numerical_cols = [
            "energy_100g", "proteins_100g", "carbohydrates_100g", 
            "sugars_100g", "fat_100g", "saturated-fat_100g",
            "fiber_100g", "salt_100g", "sodium_100g"
        ]
        summary_df = df.select(numerical_cols).summary()
        log_dataframe_show(logger, summary_df, "Summary statistics", num_rows=20)
        return df
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
