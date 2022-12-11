# Databricks notebook source
# MAGIC %md
# MAGIC # Exploring Synonymous Bigrams for LCT Keywords Using Spark
# MAGIC 
# MAGIC This notebook provides startup code for loading .xml data and further processing for this project.   
# MAGIC _<b>Make sure you are running this notebook on a cluster which use spark version 3.3.0, scala version 2.12 and has the spark-xml library (version 0.15.0) and spark-nlp (version 4.2.1) library installed, otherwise this notebook will not be able to read the data!</b>_
# MAGIC 
# MAGIC _<b>Make sure you are using DataBricks Runtime 11.3 or newer otherwise you will not be able to save any new files in this repository!</b>_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Data From DBFS

# COMMAND ----------

# MAGIC %md
# MAGIC Please place all .xml files under `DATA_PATH` (a path in DBFS that stores all xml files), and make sure no further directories are nested.

# COMMAND ----------

DATA_PATH = "/FileStore/data"

# COMMAND ----------

import os
file_list = [file.path for file in dbutils.fs.ls(DATA_PATH) if os.path.basename(file.path).endswith(".xml")]

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

# MAGIC %md
# MAGIC ## Process Dataframe With Spark-nlp Pipeline

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

# MAGIC %md
# MAGIC ## Generate Bigram Count Dictionary

# COMMAND ----------

count_res = df_bi.withColumn('bigram', f.explode(f.col("bigrams.result"))).groupBy('bigram').count().sort('count', ascending=False)

# COMMAND ----------

# dbutils.fs.rm("/FileStore/data/bigram_count.parquet", True)

# COMMAND ----------

count_res.write.parquet("/FileStore/data/bigram_count.parquet") 

# COMMAND ----------

from pyspark.sql.types import IntegerType
count_df = spark.read.parquet("/FileStore/data/bigram_count.parquet")
count_df.withColumn("count", f.col("count").cast(IntegerType()))
count_df.sort('count', ascending=False).show()

# COMMAND ----------

count_df.count()

# COMMAND ----------

count_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo: Finding Synonymous Bigram for "renewable energy"

# COMMAND ----------

count_df.filter(count_df.bigram == 'renewable energy').show()
re_num = count_df.filter(count_df.bigram == 'renewable energy').select("count").collect()[0]["count"]

# COMMAND ----------

threshold = 0.7

# COMMAND ----------

# MAGIC %md
# MAGIC `threshold` is a hyperparameter that helps filter bigram counts. Only bigrams within the range of [base_bigram_count * (1-`threshold`), base_bigram_count * (1+`threshold`)] will be selected.
# MAGIC Using larger threshold will give you more potential candidates, but requires more human effort to identify useful ones from the results.
# MAGIC Using smaller threshold will save human effort, at risk of lossing synonym candidates.

# COMMAND ----------

re_approx = count_df.filter((f.col("count") < re_num*(1+threshold)) & (f.col("count") > re_num*(1-threshold)))

# COMMAND ----------

# MAGIC %md
# MAGIC After filtering, put a core keyword(in this case, 'energy') to narrow the search.

# COMMAND ----------

re_approx.filter(f.col("bigram").contains('energy')).show(50, truncate=False)
