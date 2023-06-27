# Databricks notebook source
from databricks import feature_store

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

table_name = "prepared_patient_data"

feature_columns = [
    "ClaimCount",
    "ImmunizationCount",
    "EncounterCount",
    "MedicationCount",
    "ProcedureCount"
]

primary_keys = ["PatientId"]

features_df = spark.read.table("prepared_patient_data").select(feature_columns + primary_keys)

# COMMAND ----------

fs_table_name = "prepared_patient_data_features"

fs.create_table(
    name=fs_table_name,
    primary_keys=primary_keys,
    schema=features_df.schema,
    description="patient features"
)

fs.write_table(
    name=fs_table_name,
    df=features_df,
    mode="overwrite"
)

# COMMAND ----------


