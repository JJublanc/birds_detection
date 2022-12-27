from typing import List

import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0.anomalies_pb2 import Anomalies
from tensorflow_metadata.proto.v0.schema_pb2 import Schema
from tensorflow_metadata.proto.v0.statistics_pb2 import (
    DatasetFeatureStatisticsList,
)

from src.utils.lbc_data_validation.validation_error import DataValidationError


def check_anomalies(anomalies: Anomalies, severity_check: List = None) -> None:
    """
    Check if an DataValidationError should be raised when analysis the
    anomalies tfdv report
    :param anomalies: report of the anomalies produced by tfdv
    :param severity_check: list of severities to check in
    ["UNKNOWN", "WARNING", "ERROR"]
    :return: None
    """

    # Make sure severity are handled by the function
    if not set(severity_check).issubset({"UNKNOWN", "WARNING", "ERROR"}):
        severity_check = ["UNKNOWN", "WARNING", "ERROR"]

    # Replace severity str by severity int
    severity_check_int = []
    for element in severity_check:
        severity_check_int.append(
            ["UNKNOWN", "WARNING", "ERROR"].index(element)
        )

    if anomalies.anomaly_info:
        for col in anomalies.anomaly_info:
            anomaly_info = anomalies.anomaly_info.get(col)
            if anomaly_info.severity in severity_check_int:
                error_type = ""
                for reason in anomaly_info.reason:
                    error_type += " / " + anomaly_info.Type.Name(reason.type)
                raise DataValidationError(
                    severity=anomaly_info.severity,
                    error_type=error_type,
                    feature=col,
                    message=anomaly_info.description,
                )


def check_data_schema_anomalies_report(
    data: pd.DataFrame, schema: Schema, severity_check=None
) -> None:
    """
    Compute an tfdv schema anomalies report and check the anomalies
    :param data: data to check
    :param schema: schema of reference
    :param severity_check: list of severities to check in
    ["UNKNOWN", "WARNING", "ERROR"]
    :return: None
    """
    # Get information about data validity
    stats = tfdv.generate_statistics_from_dataframe(dataframe=data)
    anomalies = tfdv.validate_statistics(statistics=stats, schema=schema)
    check_anomalies(anomalies=anomalies, severity_check=severity_check)


def check_data_skew_anomalies_report(
    data: pd.DataFrame,
    ref_schema: Schema,
    ref_stats: DatasetFeatureStatisticsList,
    severity_check: List,
    threshold: float = 0.01,
    skew_thresholds=None,
):
    """
    Compute an tfdv skew anomalies report and check the anomalies
    :param data: data to check
    :param ref_schema: schema of reference
    :param ref_stats: stat of reference
    :param severity_check: list of severities to check in
    ["UNKNOWN", "WARNING", "ERROR"]
    :param threshold: default threshold to test the skew
    :param skew_thresholds: dict with threshold by column
    :return: None
    """

    if skew_thresholds is None:
        skew_thresholds = {}

    for col in data.columns:
        if col in skew_thresholds.keys():
            skew_thresh = skew_thresholds[col]
        else:
            skew_thresh = threshold

        # TODO make it possible to change the norm to compute distance between
        #  distributions
        tfdv.get_feature(
            ref_schema, col
        ).skew_comparator.infinity_norm.threshold = skew_thresh

        # TODO Complete with a check of the drift of the data
        stats_test = tfdv.generate_statistics_from_dataframe(dataframe=data)
        anomalies = tfdv.validate_statistics(
            statistics=ref_stats,
            schema=ref_schema,
            serving_statistics=stats_test,
        )

        check_anomalies(anomalies, severity_check=severity_check)
