import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, floor, count, avg, max
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder

def ingest_data(spark: SparkSession, file_path: str) -> DataFrame:
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    # Schema and print data to verify that the file loaded
    print("Data loaded. Schema:")
    df.printSchema()
    print("Sample of 5 rows:")
    df.show(5)
    return df


def clean_and_prepare_data(df: DataFrame) -> DataFrame:
    # drop cancelled or diverted flights
    df_filtered = df.filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0))

    # Select only columns we need (prevents data leakage)
    columns_to_keep = [
        "FL_DATE",
        "AIRLINE",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "ARR_DELAY"
    ]
    df_selected = df_filtered.select(columns_to_keep)

    print(f"Count before validation: {df_selected.count()}")

    df_validated = df_selected.filter(
        # ensuring valid time and minutes
        ((col("CRS_DEP_TIME") >= 0)&(col("CRS_DEP_TIME") <= 2359)&(col("CRS_DEP_TIME") % 100 < 60))&((col("CRS_ARR_TIME") >= 0) & (col("CRS_ARR_TIME") <= 2359)&(col("CRS_ARR_TIME") % 100 < 60))&

        # ensures distance b/w 10 - 10000 miles
        # and speeds b/w 50 to 700mph
        (col("DISTANCE") > 10) &
        (col("DISTANCE") < 10000) & 
        (col("CRS_ELAPSED_TIME") > 10) & 
        ( (col("DISTANCE") / (col("CRS_ELAPSED_TIME") / 60)) < 700 ) &
        ( (col("DISTANCE") / (col("CRS_ELAPSED_TIME") / 60)) > 50 ) & 

        # origin can't = dest
        (col("ORIGIN") != col("DEST")) & 
        
        #empty string check
        (col("AIRLINE") != "")&(col("ORIGIN") != "")&(col("DEST") != "")
    )

    # create a new dep hour feature
    df_with_target = df_validated.withColumn("Is_Delayed", when(col("ARR_DELAY") > 15, 1).otherwise(0)).withColumn("DepHour", floor(col("CRS_DEP_TIME") / 100).cast("integer"))

    final_columns = [
        "FL_DATE", "AIRLINE", "ORIGIN", "DEST", "DepHour", 
        "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE", "Is_Delayed"
    ]

    # Drop na from final set of cols
    df_cleaned = df_with_target.select(final_columns).na.drop(subset=final_columns)

    print(f"Count after validation: {df_cleaned.count()}")
    print("Cleaning complete. New schema:")
    df_cleaned.printSchema()
    df_cleaned.show(5)
    
    return df_cleaned


def run_exploratory_data_analysis(df_cleaned: DataFrame, spark: SparkSession):
    df_cleaned.createOrReplaceTempView("flights")

    # Q: What's the delay rate by day of week?
    dow_sql = """
        SELECT 
            EXTRACT(DAYOFWEEK_ISO FROM FL_DATE) AS Day_Num,
            date_format(FL_DATE, 'E') AS Day_Of_Week,
            AVG(Is_Delayed) AS Delay_Rate
        FROM flights
        GROUP BY Day_Num, Day_Of_Week
        ORDER BY Day_Num
    """
    dow_data_pd = spark.sql(dow_sql).toPandas()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=dow_data_pd, x="Day_Of_Week", y="Delay_Rate", order=dow_data_pd["Day_Of_Week"])
    plt.title("Delay rate DOW")
    plt.savefig("delay_by_dow.png")

    # Q: 10 worst most delayed routes?
    worst_routes_sql = """
        SELECT 
            CONCAT(ORIGIN, ' -> ', DEST) AS Route,
            COUNT(*) AS Total_Flights,
            AVG(Is_Delayed) AS Delay_Rate
        FROM flights
        GROUP BY Route
        HAVING Total_Flights > 500
        ORDER BY Delay_Rate DESC
        LIMIT 10
    """
    worst_routes_pd = spark.sql(worst_routes_sql).toPandas()
    plt.figure(figsize=(12, 8))
    sns.barplot(data=worst_routes_pd, y="Route", x="Delay_Rate", orient='h')
    plt.title("Top 10 Most Delay Prone Routes")
    plt.xlabel("Avg Delay Rate")
    plt.ylabel("Route")
    plt.savefig("delay_by_route.png")

    # Q: Flight delay rate by distance?
    distance_sql = """
        SELECT 
            CASE
                WHEN DISTANCE <= 250 THEN '1. (0-250 mi)'
                WHEN DISTANCE <= 500 THEN '2. (251-500 mi)'
                WHEN DISTANCE <= 750 THEN '3. (501-750 mi)'
                WHEN DISTANCE <= 1000 THEN '4. (751-1000 mi)'
                WHEN DISTANCE <= 1500 THEN '5. (1001-1500 mi)'
                WHEN DISTANCE <= 2500 THEN '6. (1501-2500 mi)'
                ELSE '7. (2501+ mi)'
            END AS Distance_Bin,
            AVG(Is_Delayed) AS Delay_Rate,
            COUNT(*) AS Total_Flights
        FROM flights
        GROUP BY Distance_Bin
        ORDER BY Distance_Bin
    """
    distance_pd = spark.sql(distance_sql).toPandas()
    plt.figure(figsize=(14, 7))
    sns.barplot(data=distance_pd, x="Distance_Bin", y="Delay_Rate")
    plt.title("Delay Rate by Flight Distance")
    plt.xlabel("Flight Distance Category")
    plt.ylabel("Average Delay Rate")
    plt.savefig("delay_by_distance.png")

    # Q: "Which airports are most prone to delays?"
    print("Top 5 Origin Airports with Highest Delay Rate:")
    origin_sql = """
        SELECT 
            ORIGIN, 
            COUNT(*) AS Total_Flights,
            AVG(Is_Delayed) AS Delay_Rate
        FROM flights
        GROUP BY ORIGIN
        HAVING Total_Flights > 1000
        ORDER BY Delay_Rate DESC
        LIMIT 5
    """
    origin_pd = spark.sql(origin_sql).toPandas()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=origin_pd, x="ORIGIN", y="Delay_Rate")
    plt.title("Top 5 Origin Airports by Delay Rate")
    plt.xlabel("Airport Code")
    plt.ylabel("Average Delay Rate")
    plt.savefig("delay_by_origin.png")

    # Q: "Which airlines have the worst delays?"
    print("Delay Rate by Airline:")
    airline_sql = """
        SELECT 
            AIRLINE, 
            COUNT(*) AS Total_Flights,
            AVG(Is_Delayed) AS Delay_Rate
        FROM flights
        GROUP BY AIRLINE
        ORDER BY Delay_Rate DESC
    """
    airline_pd = spark.sql(airline_sql).toPandas()
    plt.figure(figsize=(12, 10))
    sns.barplot(data=airline_pd, y="AIRLINE", x="Delay_Rate", orient='h')
    plt.title("Delay Rate by Airline")
    plt.xlabel("Average Delay Rate")
    plt.ylabel("Airline")
    plt.savefig("delay_by_airline.png")

    # Q: "How does delay rate change by hour of day?"
    hour_data_pd = spark.sql("SELECT DepHour, AVG(Is_Delayed) AS Delay_Rate FROM flights GROUP BY DepHour ORDER BY DepHour").toPandas()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=hour_data_pd, x="DepHour", y="Delay_Rate")
    plt.title("Flight Delay Rate by Dep Hour")
    plt.savefig("delay_by_hour.png")


def run_benchmarking(df: DataFrame, file_path: str):
    fractions_to_test = [0.1, 0.5, 1.0] # Test on 10%, 50%, and 100% of data
    benchmark_results = []
    
    for frac in fractions_to_test:
        print(f"\nBenchmarking at {frac*100}% of data")
        
        # Get a sample of the raw Spark DataFrame
        df_sample = df.sample(fraction=frac, seed=42).cache()
        df_sample.count() # Force cache
        
        # PySpark Benchmark 1 (Simple)
        print("Running PySpark Task 1 (Simple Aggregation)...")
        start_spark_1 = time.time()
        
        df_sample.filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0)) \
                 .groupBy("ORIGIN") \
                 .count() \
                 .collect()
                 
        end_spark_1 = time.time()
        time_spark_1 = end_spark_1 - start_spark_1
        print(f"PySpark Time (Task 1, {frac*100}%): {time_spark_1:.4f} seconds")
        benchmark_results.append({
            'Framework': 'PySpark', 
            'Task': 'Task 1 (Simple Agg)', 
            'Data_Size_Percent': frac * 100, 
            'Time_sec': time_spark_1
        })

        #PySpark Benchmark 2 (Complex)
        print("Running PySpark Task 2 (Complex Aggregation)...")
        start_spark_2 = time.time()

        df_sample.filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0)) \
                 .groupBy("AIRLINE") \
                 .agg(
                     count("*").alias("Total_Flights"),
                     avg("DISTANCE").alias("Avg_Distance"),
                     max("DISTANCE").alias("Max_Distance")
                 ) \
                 .collect()

        end_spark_2 = time.time()
        time_spark_2 = end_spark_2 - start_spark_2
        print(f"PySpark Time (Task 2, {frac*100}%): {time_spark_2:.4f} seconds")
        benchmark_results.append({
            'Framework': 'PySpark', 
            'Task': 'Task 2 (Complex Agg)', 
            'Data_Size_Percent': frac * 100, 
            'Time_sec': time_spark_2
        })

        #PySpark Benchmark 3 (Big Join)
        print(f"Running PySpark Task 3 (heavy join) at {frac*100}%...")
        start_spark_3 = time.time()
        try:
            df_join_sample = df_sample.filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0)) \
                                    .select("FL_DATE", "ORIGIN") \
                                    .cache()
            
            df_a = df_join_sample.alias("df_a")
            df_b = df_join_sample.alias("df_b")
            
            joined_df = df_a.join(df_b, 
              (df_a.FL_DATE == df_b.FL_DATE) & (df_a.ORIGIN == df_b.ORIGIN), 
              "inner"
            )
            joined_df.count() # Force the join to execute
            
            end_spark_3 = time.time()
            time_spark_3 = end_spark_3 - start_spark_3
            print(f"PySpark Time (Task 3, {frac*100}%): {time_spark_3:.4f} seconds")
            benchmark_results.append({
                'Framework': 'PySpark', 'Task': 'Task 3 (big Join)', 
                'Data_Size_Percent': frac * 100, 'Time_sec': time_spark_3
            })
            df_join_sample.unpersist()
        except Exception as e:
            print(f"PySpark failed (Task 3, {frac*100}%): {e}")

        #PySpark Benchmark 4 (ML Pipeline)
        print(f"Running PySpark Task 4 (ML Pipeline) at {frac*100}%...")
        start_spark_4 = time.time()
        try:
            df_ml_sample = df_sample.filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0)) \
                                    .select("AIRLINE", "ORIGIN", "DEST")
            
            categorical_cols = ["AIRLINE", "ORIGIN", "DEST"]
            indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="skip") for c in categorical_cols]
            encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec") for c in categorical_cols]
            
            pipeline = Pipeline(stages=indexers + encoders)
            model = pipeline.fit(df_ml_sample)
            model.transform(df_ml_sample).count() # Force execution
            
            end_spark_4 = time.time()
            time_spark_4 = end_spark_4 - start_spark_4
            print(f"PySpark Time (Task 4, {frac*100}%): {time_spark_4:.4f} seconds")
            benchmark_results.append({
                'Framework': 'PySpark', 'Task': 'Task 4 (ML Pipeline)', 
                'Data_Size_Percent': frac * 100, 'Time_sec': time_spark_4
            })
        except Exception as e:
            print(f"PySpark failed (Task 4, {frac*100}%): {e}")
        
        # Unpersist the main sample for this fraction
        df_sample.unpersist()


        #Pandas Benchmarks
        print(f"\nRunning Pandas Tasks ({frac*100}%)...")
        try:
            start_pandas_load = time.time()
            pandas_df_raw = pd.read_csv(
                file_path, 
                usecols=["ORIGIN", "AIRLINE", "DISTANCE", "CANCELLED", "DIVERTED", "FL_DATE"]
            )
            pandas_sample = pandas_df_raw.sample(frac=frac, random_state=42)
            del pandas_df_raw
            end_pandas_load = time.time()
            print(f"Pandas Data Load & Sample Time: {end_pandas_load - start_pandas_load:.4f} seconds")
            
            #Pandas Benchmark 1 (Simple
            start_pandas_1 = time.time()
            pandas_filtered_1 = pandas_sample[
                (pandas_sample["CANCELLED"] == 0) & (pandas_sample["DIVERTED"] == 0)
            ]
            pandas_filtered_1.groupby("ORIGIN").size()
            end_pandas_1 = time.time()
            time_pandas_1 = end_pandas_1 - start_pandas_1
            print(f"Pandas Time (Task 1, {frac*100}%): {time_pandas_1:.4f} seconds")
            benchmark_results.append({
                'Framework': 'Pandas', 'Task': 'Task 1 (Simple Agg)', 
                'Data_Size_Percent': frac * 100, 'Time_sec': time_pandas_1
            })

            #pd Benchmark 2 (Complex)
            start_pandas_2 = time.time()
            pandas_filtered_2 = pandas_sample[
                (pandas_sample["CANCELLED"] == 0) & (pandas_sample["DIVERTED"] == 0)
            ]
            pandas_filtered_2.groupby("AIRLINE")["DISTANCE"].agg(
                Total_Flights='count', Avg_Distance='mean', Max_Distance='max'
            )
            end_pandas_2 = time.time()
            time_pandas_2 = end_pandas_2 - start_pandas_2
            print(f"Pandas Time (Task 2, {frac*100}%): {time_pandas_2:.4f} seconds")
            benchmark_results.append({
                'Framework': 'Pandas', 'Task': 'Task 2 (Complex Agg)', 
                'Data_Size_Percent': frac * 100, 'Time_sec': time_pandas_2
            })
            
            #pd Benchmark 3 (heavy join)
            print(f"Running Pandas Task 3 (heavy join) at {frac*100}%...")
            start_pandas_3 = time.time()
            try:
                pandas_cleaned_3 = pandas_sample[
                    (pandas_sample["CANCELLED"] == 0) & (pandas_sample["DIVERTED"] == 0)
                ][["FL_DATE", "ORIGIN"]]
                
                pd.merge(pandas_cleaned_3, pandas_cleaned_3, on=["FL_DATE", "ORIGIN"])
                
                end_pandas_3 = time.time()
                time_pandas_3 = end_pandas_3 - start_pandas_3
                print(f"Pandas Time (Task 3, {frac*100}%): {time_pandas_3:.4f} seconds")
                benchmark_results.append({
                    'Framework': 'Pandas', 'Task': 'Task 3 (heavy join)', 
                    'Data_Size_Percent': frac * 100, 'Time_sec': time_pandas_3
                })
            except Exception as e:
                print(f"Pandas failed (Task 3, {frac*100}%): {e}")
                benchmark_results.append({
                    'Framework': 'Pandas', 'Task': 'Task 3 (heavy join)', 
                    'Data_Size_Percent': frac * 100, 'Time_sec': None
                })
            
            #pd benchmark 4 (ML Pipeline)
            print(f"Running Pandas Task 4 (ML Pipeline) at {frac*100}%...")
            start_pandas_4 = time.time()
            try:
                pandas_ml_sample = pandas_sample[["AIRLINE", "ORIGIN", "DEST"]]
                pd.get_dummies(pandas_ml_sample, columns=["AIRLINE", "ORIGIN", "DEST"])
                
                end_pandas_4 = time.time()
                time_pandas_4 = end_pandas_4 - start_pandas_4
                print(f"Pandas Time (Task 4, {frac*100}%): {time_pandas_4:.4f} seconds")
                benchmark_results.append({
                    'Framework': 'Pandas', 'Task': 'Task 4 (ML Pipeline)', 
                    'Data_Size_Percent': frac * 100, 'Time_sec': time_pandas_4
                })
            except Exception as e:
                print(f"Pandas failed (Task 4, {frac*100}%): {e}")
                benchmark_results.append({
                    'Framework': 'Pandas', 'Task': 'Task 4 (ML Pipeline)', 
                    'Data_Size_Percent': frac * 100, 'Time_sec': None
                })
            
            del pandas_sample # Free memory

        except Exception as e:
            print(f"Pandas failed at {frac*100}%: {e}")
            # Add nulls for all pandas tasks if the initial load failed
            if 'pandas_sample' not in locals():
                for i in range(1, 5):
                    benchmark_results.append({
                        'Framework': 'Pandas', f'Task': 'Task {i}', 
                        'Data_Size_Percent': frac * 100, 'Time_sec': None
                    })

    print("\nFinal Benchmark Results")
    results_df = pd.DataFrame(benchmark_results)
    print(results_df.to_string())# to make sure whole table prints

    # Create a plot
    print("\nSaving benchmark plot...")
    plt.figure(figsize=(14, 7))
    g = sns.catplot(
        data=results_df, 
        x='Data_Size_Percent', 
        y='Time_sec', 
        hue='Framework', 
        col='Task', 
        kind='bar',
        sharey=False,  
        col_wrap=2   
    )
    g.fig.suptitle('Pandas vs. PySpark Performance Benchmark', y=1.03)
    g.set_axis_labels("Data Size (%)", "Time (seconds)")
    plt.savefig("benchmark_results.png", bbox_inches='tight')
    plt.close()
    print("Benchmark plot saved as benchmark_results.png")

def run_feature_transformation(df_cleaned: DataFrame) -> DataFrame:
    print("\nFeature Transformation")

    # cols to transform
    categorical_cols = ["AIRLINE", "ORIGIN", "DEST"]

    # This bit will converts strings to numbers "JFK" -> 0.0  etc
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="skip") 
        for c in categorical_cols
    ]

    #Here one hot encoder converst nums into vectors
    encoder_inputs = [f"{c}_idx" for c in categorical_cols]
    encoder_outputs = [f"{c}_vec" for c in categorical_cols]
    
    encoders = [
        OneHotEncoder(inputCols=encoder_inputs, outputCols=encoder_outputs)
    ]

    #create pipeline and fit to data then transform
    pipeline = Pipeline(stages=indexers + encoders)
    feature_model = pipeline.fit(df_cleaned)
    df_transformed = feature_model.transform(df_cleaned)

    print("Feature transformation complete. New cols added:")
    df_transformed.printSchema()
    
    print("\nSample of transformed data:")
    df_transformed.select("AIRLINE", "AIRLINE_idx", "AIRLINE_vec", "ORIGIN", "ORIGIN_idx", "ORIGIN_vec").show(5, truncate=False)
    
    return df_transformed

def main():
    file_path = "flight_dataset.csv"
    spark = None
    try:
        # Initialize Spark
        spark = SparkSession.builder.appName("FlightDelayPredictor").config("spark.driver.memory", "4g").getOrCreate()
        
        # Run the 4 pipeline steps
        df_raw = ingest_data(spark, file_path)
        df_cleaned = clean_and_prepare_data(df_raw)
        run_exploratory_data_analysis(df_cleaned, spark)
        run_benchmarking(df_raw, file_path)
        run_feature_transformation(df_cleaned)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if spark:
            print("Stopping Spark session.")
            spark.stop()


if __name__ == "__main__":
    main()