# Databricks notebook source
dbutils.widgets.text("experiment_name", "patient-cost-model")
experiment_name = dbutils.widgets.get("experiment_name")

dbutils.widgets.text("model_name", "patient-cost-model")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

import mlflow 

experiment = mlflow.get_experiment_by_name(f"/Shared/{experiment_name}")

client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment.experiment_id, "", order_by=["metrics.test_root_mean_squared_error DESC"], max_results=1)
best_run = runs[0]

# COMMAND ----------

import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# Wait until the model is ready
def wait_until_ready(model_name, model_version):
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)

# The default path where the MLflow autologging function stores the TensorFlow Keras model
artifact_path = "model"
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=best_run.info.run_id, artifact_path=artifact_path)

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
wait_until_ready(model_details.name, model_details.version)
