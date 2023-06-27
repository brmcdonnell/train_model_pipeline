# Databricks notebook source
# MAGIC %md
# MAGIC # Re-training Pipeline

# COMMAND ----------

dbutils.widgets.text("experiment_name", "patient-cost-model")
experiment_name = dbutils.widgets.get("experiment_name")

dbutils.widgets.text("n_trials", "32")
n_trials = int(dbutils.widgets.get("n_trials"))

dbutils.widgets.text("feature_table", "prepared_patient_data_features")
feature_table_name = dbutils.widgets.get("feature_table")

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import FeatureLookup
from sklearn.model_selection import train_test_split
import pyspark.sql.functions as F

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

primary_key = "PatientId"
# In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
model_feature_lookups = [FeatureLookup(table_name=feature_table_name, lookup_key=primary_key)]

target_column = "Healthcare_Coverage"
labels = spark.read.table("patients").select([F.col("ID").alias(primary_key), target_column])

# fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
training_set = fs.create_training_set(labels, model_feature_lookups, label=target_column)
training_pd = training_set.load_df().toPandas()

# Create train and test datasets
X = training_pd.drop([primary_key, target_column], axis=1)
y = training_pd[target_column]

# Prepare training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# COMMAND ----------

import mlflow
import databricks.automl_runtime

# Create and/or set the experiment
try:
    mlflow.create_experiment(f"/Shared/{experiment_name}")
except:
    pass
mlflow.set_experiment(f"/Shared/{experiment_name}")

# Set autologging for sklearn
mlflow.xgboost.autolog(log_input_examples=True, silent=True, log_models=True)

# COMMAND ----------

import pandas as pd
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import sklearn
from xgboost import XGBRegressor
from hyperopt import hp, tpe, fmin, STATUS_OK, SparkTrials


def objective(params):
    with mlflow.start_run() as mlflow_run:
        model = XGBRegressor(**params)
        model = model.fit(X_train, y_train, early_stopping_rounds=5, verbose=False, eval_set=[(X_val, y_val)])

        # Log metrics for the training set
        signature = mlflow.xgboost.infer_signature(X_train, y_train)
        model_info = mlflow.xgboost.log_model(model, "model", signature=signature)

        training_eval_result = mlflow.evaluate(
            model=model_info.model_uri,
            data=X_train.assign(**{str(target_column):y_train}),
            targets=target_column,
            model_type="regressor",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "train_"}
        )
        # Log metrics for the validation set
        val_eval_result = mlflow.evaluate(
            model=model_info.model_uri,
            data=X_val.assign(**{str(target_column):y_val}),
            targets=target_column,
            model_type="regressor",
            evaluator_config= {"log_model_explainability": False,
                                "metric_prefix": "val_"}
        )
        xgb_val_metrics = val_eval_result.metrics
        # Log metrics for the test set
        test_eval_result = mlflow.evaluate(
            model=model_info.model_uri,
            data=X_test.assign(**{str(target_column):y_test}),
            targets=target_column,
            model_type="regressor",
            evaluator_config= {"log_model_explainability": False,
                                "metric_prefix": "test_"}
        )
        xgb_test_metrics = test_eval_result.metrics

        loss = xgb_val_metrics["val_root_mean_squared_error"]

        # Truncate metric key names so they can be displayed together
        xgb_val_metrics = {k.replace("val_", ""): v for k, v in xgb_val_metrics.items()}
        xgb_test_metrics = {k.replace("test_", ""): v for k, v in xgb_test_metrics.items()}

        return {
            "loss": loss,
            "status": STATUS_OK,
            "val_metrics": xgb_val_metrics,
            "test_metrics": xgb_test_metrics,
            "model": model,
            "run": mlflow_run,
        }

# COMMAND ----------

# Conduct the hyperparameter search using Spark for the individual trials
hyperparameter_search_space = {
        'n_estimators': hp.uniformint('n_estimators', 5, 100),
        'max_leaves': hp.choice("max_leaves", [None, hp.uniformint('max_leaves_int', 5, 100)]),
        'max_depth': hp.choice('max_depth', [3, 4, 5]),
}

trials = SparkTrials()
fmin(objective,
     space=hyperparameter_search_space,
     algo=tpe.suggest,
     max_evals=n_trials,
     trials=trials)
