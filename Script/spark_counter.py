# Databricks notebook source
# MAGIC %md
# MAGIC # Exploring reddit data using Spark
# MAGIC 
# MAGIC This notebook provides startup code for downloading Reddit data from the Azure Blob Storage bucket specifically setup for this project.   
# MAGIC _<b>Make sure you are running this notebook on a cluster which has the credentials setup to access Azure Blob Storage, otherwise this notebook will not be able to read the data!</b>_
# MAGIC 
# MAGIC _<b>Make sure you are using DataBricks Runtime 11.3 or newer otherwise you will not be able to save any new files in this repository!</b>_
# MAGIC 
# MAGIC The dataset for this notebook is described in [The Pushshift Reddit Dataset](https://arxiv.org/pdf/2001.08435.pdf) paper.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Data From DBFS

# COMMAND ----------

import os
file_list = [file.path for file in dbutils.fs.ls("/FileStore/data") if os.path.basename(file.path).endswith(".xml")]

# COMMAND ----------

file_list

# COMMAND ----------

df_raw=spark.read.format('com.databricks.spark.xml').options(rowTag='Job').load(','.join(file_list))
#df_raw=spark.read.format('com.databricks.spark.xml').options(rowTag='Job').load("/FileStore/data/US_XML_AddFeed_20100101_20100107.xml")
df_raw.show()

# COMMAND ----------

df_raw.count()

# COMMAND ----------

df_raw.printSchema()

# COMMAND ----------

df_raw.select("JobText").show(truncate=False)

# COMMAND ----------

# dbutils.fs.rm("/FileStore/data/US_XML_AddFeed_20100101_20100107.xml", True)

# COMMAND ----------

import pyspark.sql.functions as f
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

# COMMAND ----------

document_assembler = DocumentAssembler() \
    .setInputCol("JobText")
    
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

bigrams = NGramGenerator() \
            .setInputCols(["token"]) \
            .setOutputCol("bigrams") \
            .setN(2) 

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer, 
    bigrams,
])

# COMMAND ----------

model = pipeline.fit(df_raw)
df_bi = model.transform(df_raw)

# COMMAND ----------

df_bi.select("bigrams.result").show(2, truncate=200)

# COMMAND ----------

count_res = df_bi.withColumn('bigram', f.explode(f.col("bigrams.result"))).groupBy('bigram').count().sort('count', ascending=False)
count_res.show(truncate=False)

# COMMAND ----------

count_res.count()

# COMMAND ----------

count_res.write.parquet("/FileStore/data/bigram_count.parquet") 

# COMMAND ----------

from pyspark.sql.types import IntegerType
count_df = spark.read.parquet("/FileStore/data/bigram_count.parquet")
count_df.withColumn("count", f.col("count").cast(IntegerType()))
count_df.sort('count', ascending=False).show()

# COMMAND ----------

count_df.printSchema()

# COMMAND ----------

count_df.filter(count_df.bigram == 'renewable energy').show()

# COMMAND ----------

re_approx = count_df.filter((f.col("count") < 34) & (f.col("count") > 22))

# COMMAND ----------

re_approx.filter(f.col("bigram").contains('energy')).show(truncate=False)

# COMMAND ----------

from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
import pyspark.sql.functions as f
from pyspark.sql.functions import when, col
from pyspark.ml.feature import CountVectorizer, IDF, HashingTF, SQLTransformer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sparknlp.pretrained import PretrainedPipeline
from pyspark.ml.linalg import Vectors
# Start Spark Session with Spark NLP
# spark = sparknlp.start()
