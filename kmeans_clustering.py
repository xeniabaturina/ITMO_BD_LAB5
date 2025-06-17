import os
import numpy as np
import matplotlib.pyplot as plt
import findspark
findspark.init()

from tqdm import tqdm

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA
from pyspark.sql.functions import col, avg
import pandas as pd

from config import (
    FEATURE_CONFIG, 
    PROCESSED_DATA_PATH, KMEANS_MODEL_PATH, REPORT_PATH, PLOTS_DIR,
    get_logger, log_dataframe_show
)
from spark_setup import create_spark_session

logger = get_logger(__name__)


def prepare_features(df):
    """Prepare features for clustering"""
    logger.info("Preparing features for clustering...")

    feature_cols = FEATURE_CONFIG["feature_cols"]

    filtered_df = df
    for col_name in feature_cols:
        percentiles = filtered_df.approxQuantile(
            col_name, 
            FEATURE_CONFIG["outlier_percentile"],
            FEATURE_CONFIG["outlier_approx_error"]
        )
        lower_bound, upper_bound = percentiles[0], percentiles[1]
        filtered_df = filtered_df.filter(
            (col(col_name) >= lower_bound) & (col(col_name) <= upper_bound)
        )

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    scaler = StandardScaler(
        inputCol="features", 
        outputCol="scaled_features",
        withStd=True, 
        withMean=True
    )

    pipeline = Pipeline(stages=[assembler, scaler])

    pipeline_model = pipeline.fit(filtered_df)
    prepared_df = pipeline_model.transform(filtered_df)
    
    logger.info(f"Features prepared. Filtered dataset has {prepared_df.count()} records.")
    
    return prepared_df, pipeline_model, feature_cols

def find_optimal_k(prepared_df, max_k=15):
    """Find optimal k using both WSSSE and silhouette scores."""
    logger.info("Finding optimal number of clusters...")

    costs = []
    silhouettes = []

    evaluator = ClusteringEvaluator(
        predictionCol='prediction',
        featuresCol='scaled_features',
        metricName='silhouette'
    )

    with tqdm(range(2, max_k + 1), desc="Evaluating k", unit="clusters") as pbar:
        for k in pbar:
            pbar.set_description(f"Evaluating k={k}")

            kmeans = KMeans(
                k=k, 
                seed=42, 
                featuresCol="scaled_features", 
                maxIter=50
            )

            model = kmeans.fit(prepared_df)

            cost = model.summary.trainingCost
            costs.append(cost)

            predictions = model.transform(prepared_df)
            silhouette = evaluator.evaluate(predictions)
            silhouettes.append(silhouette)
            
            pbar.set_postfix(cost=f"{cost:.2f}", silhouette=f"{silhouette:.4f}")
            logger.info(f"k={k}, WSSSE={cost:.2f}, Silhouette Score={silhouette:.4f}")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(range(2, max_k + 1), costs, marker='o')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Cost (WSSSE)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)

    ax2.plot(range(2, max_k + 1), silhouettes, marker='o')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Scores for Different k')
    ax2.grid(True)
    
    plt.tight_layout()

    metrics_plot_path = os.path.join(PLOTS_DIR, "clustering_metrics.png")
    plt.savefig(metrics_plot_path)
    plt.close()
    
    logger.info(f"Clustering metrics plot saved to {metrics_plot_path}")

    optimal_k_silhouette = silhouettes.index(max(silhouettes)) + 2

    diffs = np.diff(costs)
    optimal_k_elbow = np.argmin(diffs) + 2
    
    logger.info(f"Optimal k based on silhouette score: {optimal_k_silhouette}")
    logger.info(f"Optimal k based on elbow method: {optimal_k_elbow}")

    optimal_k = optimal_k_silhouette
    logger.info(f"Using optimal k = {optimal_k}")
    
    return optimal_k

def build_kmeans_model(prepared_df, k):
    """Build a K-means clustering model with improved parameters."""
    logger.info(f"Building K-means model with k={k}...")

    kmeans = KMeans(
        k=k, 
        seed=42, 
        featuresCol="scaled_features",
        maxIter=200,
        tol=1e-6,
        initSteps=10
    )
    
    model = kmeans.fit(prepared_df)
    
    predictions = model.transform(prepared_df)
    
    os.makedirs(os.path.dirname(KMEANS_MODEL_PATH), exist_ok=True)
    
    if os.path.exists(KMEANS_MODEL_PATH):
        logger.info(f"Model directory {KMEANS_MODEL_PATH} already exists. Overwrite...")
        model.write().overwrite().save(KMEANS_MODEL_PATH)
    else:
        model.save(KMEANS_MODEL_PATH)
        
    logger.info(f"Model saved to {KMEANS_MODEL_PATH}")
    
    return model, predictions

def evaluate_model(predictions):
    """Evaluate the clustering model."""
    logger.info("Evaluating model...")

    evaluator = ClusteringEvaluator(
        predictionCol='prediction',
        featuresCol='scaled_features',
        metricName='silhouette'
    )

    silhouette = evaluator.evaluate(predictions)
    logger.info(f"Silhouette score: {silhouette:.2f}")
    
    return silhouette

def analyze_clusters(predictions, feature_cols, k):
    """Analyze the clusters to understand their characteristics."""
    logger.info("Analyzing clusters...")

    agg_exprs = [avg(c).alias(f"avg_{c}") for c in feature_cols]

    cluster_stats = predictions.groupBy("prediction").agg(*agg_exprs)

    from pyspark.sql.functions import format_number

    formatted_stats = cluster_stats
    for col_name in [f"avg_{c}" for c in feature_cols]:
        formatted_stats = formatted_stats.withColumn(
            col_name, 
            format_number(col_name, 2)
        )

    log_dataframe_show(logger, formatted_stats, "Cluster statistics", num_rows=20)

    cluster_sizes = predictions.groupBy("prediction").count().orderBy("prediction")
    log_dataframe_show(logger, cluster_sizes, "Cluster sizes", num_rows=20)

    cluster_stats_pd = cluster_stats.toPandas()

    for col in cluster_stats_pd.columns:
        if col != 'prediction':
            cluster_stats_pd[col] = cluster_stats_pd[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (float, np.float64)) else x)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, col_name in enumerate(feature_cols):
        avg_col_name = f"avg_{col_name}"
        ax = axes[i]

        plot_data = cluster_stats_pd.copy()
        plot_data[avg_col_name] = pd.to_numeric(plot_data[avg_col_name], errors='coerce')
        
        plot_data.plot(
            kind='bar',
            x='prediction',
            y=avg_col_name,
            ax=ax,
            color='skyblue',
            edgecolor='black'
        )
        ax.set_title(f'Average {col_name} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(f'Average {col_name}')
        ax.grid(True, alpha=0.3)

        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    cluster_stats_plot_path = os.path.join(PLOTS_DIR, "cluster_statistics.png")
    plt.savefig(cluster_stats_plot_path)
    plt.close()
    
    logger.info(f"Cluster statistics plot saved to {cluster_stats_plot_path}")

    plt.figure(figsize=(18, 10))

    cluster_heatmap_data = cluster_stats.toPandas().set_index('prediction')

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(cluster_heatmap_data)
    normalized_df = pd.DataFrame(
        normalized_data, 
        index=cluster_heatmap_data.index, 
        columns=cluster_heatmap_data.columns
    )

    import seaborn as sns
    sns.heatmap(
        normalized_df, 
        annot=True, 
        cmap='viridis', 
        fmt='.2f',
        linewidths=0.5
    )
    plt.title('Normalized Cluster Characteristics')
    plt.ylabel('Cluster')
    plt.tight_layout()

    heatmap_path = os.path.join(PLOTS_DIR, "cluster_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    
    logger.info(f"Cluster heatmap saved to {heatmap_path}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write("K-means Clustering Report\n")
        f.write("========================\n\n")
        f.write(f"Number of clusters (k): {k}\n")
        f.write(f"Silhouette score: {evaluate_model(predictions):.2f}\n\n")
        f.write("Cluster Statistics:\n")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        f.write(str(cluster_stats_pd))
        f.write("\n\nCluster Sizes:\n")
        f.write(str(predictions.groupBy("prediction").count().orderBy("prediction").toPandas()))
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
    
    logger.info(f"Clustering report saved to {REPORT_PATH}")
    
    return cluster_stats

def visualize_clusters_pca(predictions):
    """Visualize the clusters using PCA for dimensionality reduction."""
    logger.info("Visualizing clusters using PCA...")

    pca = PCA(
        k=2,
        inputCol="scaled_features",
        outputCol="pca_features"
    )

    pca_model = pca.fit(predictions)
    pca_result = pca_model.transform(predictions)

    pdf = pca_result.select("pca_features", "prediction").toPandas()

    pdf["x"] = pdf["pca_features"].apply(lambda x: float(x[0]))
    pdf["y"] = pdf["pca_features"].apply(lambda x: float(x[1]))

    plt.figure(figsize=(12, 10))
    
    n_clusters = pdf["prediction"].nunique()
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster in range(n_clusters):
        cluster_points = pdf[pdf["prediction"] == cluster]
        plt.scatter(
            cluster_points["x"],
            cluster_points["y"],
            s=50,
            alpha=0.7,
            c=[colors[cluster]],
            label=f"Cluster {cluster}"
        )
    
    plt.title("PCA Cluster Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    pca_plot_path = os.path.join(PLOTS_DIR, "pca_clusters.png")
    plt.savefig(pca_plot_path)
    plt.close()
    
    logger.info(f"PCA cluster visualization saved to {pca_plot_path}")

def main():
    spark = create_spark_session()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            logger.error(f"Processed data not found at {PROCESSED_DATA_PATH}")
            logger.error("Please run data_processor.py first to prepare the dataset.")
            return
        
        logger.info(f"Loading processed data from {PROCESSED_DATA_PATH}...")
        df = spark.read.parquet(PROCESSED_DATA_PATH)
        
        logger.info("Sample of processed data:")
        
        from pyspark.sql.functions import format_number
        
        numeric_cols = [
            "energy_100g", "proteins_100g", "carbohydrates_100g", 
            "sugars_100g", "fat_100g", "saturated-fat_100g",
            "fiber_100g", "salt_100g", "sodium_100g"
        ]
        
        display_df = df
        for col_name in numeric_cols:
            if col_name in df.columns:
                display_df = display_df.withColumn(
                    col_name, 
                    format_number(col_name, 2)
                )
        
        log_dataframe_show(logger, display_df, "Sample of processed data", num_rows=5)
        
        prepared_df, pipeline_model, feature_cols = prepare_features(df)
        
        k = find_optimal_k(prepared_df)
        
        # Override k if we get too few meaningful clusters
        if k <= 3:
            logger.info(f"Optimal k={k} may result in too few clusters. Using k=4 for better food categorization.")
            k = 4
        
        model, predictions = build_kmeans_model(prepared_df, k)
        
        silhouette = evaluate_model(predictions)
        
        cluster_stats = analyze_clusters(predictions, feature_cols, k)
        
        visualize_clusters_pca(predictions)
        
        logger.info("K-means clustering completed successfully!")
        logger.info(f"Model saved to {KMEANS_MODEL_PATH}")
        logger.info(f"Reports and visualizations saved to {PLOTS_DIR}")
        
        return model, predictions, silhouette
    
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
