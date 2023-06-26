# Databricks notebook source
# MAGIC %md
# MAGIC # Re-training Pipeline

# COMMAND ----------

dbutils.widgets.text("experiment_name", "patient-cost-model")
experiment_name = dbutils.widgets.get("experiment_name")

dbutils.widgets.text("n_trials", "32")
n_trials = int(dbutils.widgets.get("n_trials"))

# COMMAND ----------

from sklearn.model_selection import train_test_split

feature_columns = [
    "ClaimCount",
    "ImmunizationCount",
    "EncounterCount",
    "MedicationCount",
    "ProcedureCount"
]
target_column = "Healthcare_Coverage"

data = spark.read.table("prepared_patient_data").select(feature_columns + [target_column]).toPandas()

# Prepare training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(data[feature_columns], data[target_column], test_size=0.6, random_state=42)
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
