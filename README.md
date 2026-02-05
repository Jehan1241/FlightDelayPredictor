**Distributed ML Pipeline for Flight Delay Prediction**

**Project Overview**

This project implements a scalable, distributed machine learning system to predict domestic U.S. flight delays. Using a dataset of 1M+ flight records, the pipeline demonstrates the necessity of distributed computing for tasks that trigger memory "spilling" or extreme latency in traditional single-node environments (like Pandas).

**Key Technical Features**
Distributed Architecture: Leveraged PySpark’s Lazy Evaluation and the Catalyst Optimizer to handle data ingestion and preprocessing across a distributed cluster.

Feature Engineering at Scale: * StringIndexing & One-Hot Encoding (OHE): Transformed high-cardinality categorical data (Airlines, Origin/Destination airports) into sparse vectors.

VectorAssembler: Consolidated features into a single feature vector for efficient processing by MLlib algorithms.

Model Optimization: * Implemented Gradient Boosted Trees (GBT), achieving a final AUC of 0.6535.

Resolved severe class imbalance via downsampling.

Utilized Grid Search for distributed hyperparameter tuning.

**Tech Stack**
- Core Engine: Apache Spark / PySpark
- ML Libraries: Spark MLlib (GBT, Logistic Regression, Factorization Machines)
- Data Handling: SQL, Parquet, CSV
- Environment: Databricks / Local Spark Cluster

