import os
from typing import Any, Union

import mlflow
import pandas as pd
import tensorflow as tf
from mlflow.tracking import MlflowClient

from src.utils.lbc_azure_ml.azure_connection import (
    get_service_principal_auth,
    get_workspace,
)


def get_mlflow_azure_client(
    service_principal_file_name: str, workspace_file_name: str
) -> MlflowClient:
    """
    Get a mlflow client linked to the worskspace Azure so that mlflow
    :param service_principal_file_name: name used to identify the json file
    with service principal credentials
    :param workspace_file_name: name used to identify the json file
    with workspace
    :return
    a object client with which one can perform several operation concerning
    model tracking with mlflow
    """
    sp = get_service_principal_auth(service_principal_file_name)
    ws = get_workspace(workspace_file_name, sp)
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    client = MlflowClient()

    return client


def get_experiment_id(exp_name: str, client: MlflowClient) -> Union[str, None]:
    """
    From a name get the experiment id in mlflow
    :param exp_name: name of the experiment used when training the models
    :param client: MlfloClient
    :return: the experiment id
    """
    for exp in client.list_experiments():
        if exp.name == exp_name:
            return exp.experiment_id
        else:
            return None


def get_runs(experiment_name: str, client: MlflowClient) -> pd.DataFrame:
    """
    Get all runs information for a given experiment
    :param experiment_name:
    :param client:
    :return:
    a dataframe with run ids and metrics information
    """
    experiment_id = get_experiment_id(experiment_name, client)

    runs = client.search_runs(experiment_ids=[experiment_id])

    runs_df = pd.DataFrame()
    for run in runs:
        dict_run = run.to_dictionary()["data"]["metrics"]
        dict_run["run_id"] = run.to_dictionary()["info"]["run_id"]
        runs_df = runs_df.append(dict_run, ignore_index=True)

    return runs_df


def download_best_model(
    metric: str, dst_path: str, runs_df: pd.DataFrame, client: MlflowClient
):
    """
    Download locally the best model from the mlflow tracking repo.
    :param metric: metric used classify models
    :param dst_path: path where to copy the artefact
    :param runs_df: dataframe containing runs ids and results
    :param client: a MlflowClient
    :return: the path where the model is downloaded
    """

    # Get the best model id
    try:
        index = runs_df[metric].argmax()
    except KeyError:
        if runs_df.empty:
            raise KeyError(
                "No runs found. The dataframe is empty. Make sure "
                "you used the good credentials, names etc."
            )
        else:
            raise KeyError("Make sure you use a valable metric name")

    best_run_id = runs_df.loc[index, "run_id"]

    # Load the model
    os.makedirs(dst_path, exist_ok=True)
    client.download_artifacts(
        run_id=best_run_id, path="model", dst_path=dst_path
    )

    return dst_path


def load_tf_keras_model(path: str) -> Any:
    """
    If the model is a tensorflow keras model, use this function to load it
    :param path: where is saved the model folders and files
    :return: the tf.keras model
    """
    model = tf.keras.models.load_model(f"{path}/model/data/model")
    return model
