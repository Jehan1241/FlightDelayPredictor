import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, floor

# 1. Data Ingestion
file_path = "flight_dataset.csv"
spark = SparkSession.builder.appName("FlightDelayPredictor").getOrCreate()
df = spark.read.csv(file_path, header=True, inferSchema=True)

#Schema and print data to verify that the file loaded 
print("Data loaded. Schema:")
df.printSchema()
print("Sample of 5 rows:")
df.show(5)


# 2. Initial Cleaning & Target Creation

#Remove cancelled and delayed flights
df_filtered = df.filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0))

#These are the features we will use for training the model other features like DEP_TIME and DEP_DELAY
#Will be removed as they are leaky and will mess with the training of the model
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
   # 1. Time Validation
    ((col("CRS_DEP_TIME") >= 0)&(col("CRS_DEP_TIME") <= 2359)&(col("CRS_DEP_TIME") % 100 < 60))&((col("CRS_ARR_TIME") >= 0) & (col("CRS_ARR_TIME") <= 2359)&(col("CRS_ARR_TIME") % 100 < 60))&

    # 2. Physical Validation
    (col("DISTANCE") > 10) &                                        # Min distance > 10 miles
    (col("DISTANCE") < 10000) &                                     # Max distance < 10,000 miles (longest flight is ~9.5k)
    (col("CRS_ELAPSED_TIME") > 10) &                                # Min scheduled time > 10 minutes
    ( (col("DISTANCE") / (col("CRS_ELAPSED_TIME") / 60)) < 700 ) &  # checking for impossible speeds max 700mph
    ( (col("DISTANCE") / (col("CRS_ELAPSED_TIME") / 60)) > 50 ) &   # min 50mph

    # 3. Logical Validation
    (col("ORIGIN") != col("DEST")) &                                 # Origin and Destination can't be the same
    
    # 4. Empty string check
    (col("AIRLINE") != "")&(col("ORIGIN") != "")&(col("DEST") != "")
)


# If delay > 15 mins, mark delayed
# DepHour holds the hour of the day
df_with_target = df_validated.withColumn("Is_Delayed", when(col("ARR_DELAY") > 15, 1).otherwise(0)).withColumn("DepHour", floor(col("CRS_DEP_TIME") / 100).cast("integer"))

final_columns = [
    "FL_DATE", "AIRLINE", "ORIGIN", "DEST", "DepHour", 
    "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE", "Is_Delayed"
]

#drop null values
df_cleaned = df_with_target.select(final_columns).na.drop(subset=final_columns)

print(f"Count after validation: {df_cleaned.count()}")


print("Cleaning complete. New schema:")
df_cleaned.printSchema()
df_cleaned.show(5)


#3. EDA
### Needs to be expanded
df_cleaned.createOrReplaceTempView("flights")

# Q: "Which airports are most prone to delays?"
print("Top 5 Origin Airports with Highest Delay Rate:")
spark.sql("""
    SELECT 
        ORIGIN, 
        COUNT(*) AS Total_Flights,
        AVG(Is_Delayed) AS Delay_Rate
    FROM flights
    GROUP BY ORIGIN
    HAVING Total_Flights > 1000 -- Only look at airports with decent traffic
    ORDER BY Delay_Rate DESC
""").show(5)



# Q: "Which airlines have the worst delays?"
print("Delay Rate by Airline:")
spark.sql("""
    SELECT 
        AIRLINE, 
        COUNT(*) AS Total_Flights,
        AVG(Is_Delayed) AS Delay_Rate
    FROM flights
    GROUP BY AIRLINE
    ORDER BY Delay_Rate DESC
""").show()


# Q: "How does delay rate change by hour of day?"
print("Delay Rate by Departure Hour:")
spark.sql("""
    SELECT 
        DepHour, 
        COUNT(*) AS Total_Flights,
        AVG(Is_Delayed) AS Delay_Rate
    FROM flights
    GROUP BY DepHour
    ORDER BY DepHour
""").show()


# Get the "delay by hour" data
hour_data_pd = spark.sql("SELECT DepHour, AVG(Is_Delayed) AS Delay_Rate FROM flights GROUP BY DepHour ORDER BY DepHour").toPandas()

# Plot it
plt.figure(figsize=(12, 6))
sns.barplot(data=hour_data_pd, x="DepHour", y="Delay_Rate")
plt.title("Flight Delay Rate by Departure Hour")
plt.savefig("delay_by_hour.png") # <-- THIS IS THE FIX
print("Plot saved as delay_by_hour.png")

# And plot more?





#4. Benchmarking
# should probably also be expnded

# --- PySpark Benchmark ---
print("Running PySpark aggregation...")
start_spark = time.time()


df.filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0)) \
  .groupBy("ORIGIN") \
  .count() \
  .collect()

end_spark = time.time()
print(f"PySpark Time: {end_spark - start_spark:.4f} seconds")


# --- Pandas Benchmark ---
print("Running Pandas aggregation...")
start_pandas = time.time()

try:
    pandas_df = pd.read_csv(
        file_path, 
        usecols=["ORIGIN", "CANCELLED", "DIVERTED"]
    )

    pandas_df_filtered = pandas_df[
        (pandas_df["CANCELLED"] == 0) & (pandas_df["DIVERTED"] == 0)
    ]

    pandas_df_filtered.groupby("ORIGIN").size()
    
    end_pandas = time.time()
    print(f"Pandas Time: {end_pandas - start_pandas:.4f} seconds")

except Exception as e:
    print(f"Pandas failed: {e}")