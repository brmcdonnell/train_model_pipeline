# Databricks notebook source
# Retrieve notebook parameters
dbutils.widgets.text("model_name", "patient-cost-model")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# MAGIC %md 
# MAGIC # Load the latest registered model

# COMMAND ----------

import mlflow 

client = mlflow.tracking.MlflowClient()
registered_model = client.get_registered_model(model_name)
model_uri = f"models:/{model_name}/{registered_model.latest_versions[-1].version}"
model = mlflow.xgboost.load_model(model_uri)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Retrieve data to score
# MAGIC

# COMMAND ----------

feature_columns = [
    "ClaimCount",
    "ImmunizationCount",
    "EncounterCount",
    "MedicationCount",
    "ProcedureCount"
]

data = spark.read.table("prepared_patient_data").select(feature_columns).toPandas()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Score the model and save the result

# COMMAND ----------

predictions = model.predict(data)
data["prediction"] = predictions

spark.createDataFrame(data).write.mode("overwrite").saveAsTable("model_predictions")
