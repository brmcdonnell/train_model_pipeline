# Databricks notebook source
# Get the name of the table we want to load
dbutils.widgets.text(name="table_name", defaultValue="claims")
table_name = dbutils.widgets.get("table_name")

dbutils.widgets.text(name="sas_token", defaultValue="")
sas_token = dbutils.widgets.get("sas_token")

# COMMAND ----------

# Settings to connect to Azure storage account blob storage
spark.conf.set("fs.azure.account.auth.type.edsdbpocdata.dfs.core.windows.net", "SAS")
spark.conf.set("fs.azure.sas.token.provider.type.edsdbpocdata.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
spark.conf.set("fs.azure.sas.fixed.token.edsdbpocdata.dfs.core.windows.net", sas_token)

# COMMAND ----------

spark.sql(f"drop table if exists {table_name}").collect()
df = spark.read.csv(f"abfs://data@edsdbpocdata.dfs.core.windows.net/NewMexicoData/HcscSample/csv/{table_name}.csv", header=True, inferSchema=True)
df.write.mode("overwrite").saveAsTable(table_name)
