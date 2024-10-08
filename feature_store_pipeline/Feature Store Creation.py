# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

# Databricks notebook source
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

# COMMAND ----------

fe = FeatureEngineeringClient()

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

fe.create_table(
    name=fs_table_name,
    primary_keys=primary_keys,
    schema=features_df.schema,
    description="patient features"
)

fe.write_table(
    name=fs_table_name,
    df=features_df,
    mode="overwrite"
)

# COMMAND ----------


