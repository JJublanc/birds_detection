from typing import Tuple

import pandas as pd
import tensorflow_data_validation as tfdv

from src.utils.lbc_data_validation.tfdv_validation import (
    check_data_schema_anomalies_report,
    check_data_skew_anomalies_report,
)

# https://www.tensorflow.org/tfx/data_validation/get_started
# https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/anomalies.proto

DATA_PATH = "./data/DailyDelhiClimate.csv"


def get_fake_data(
    path: str, n: int, data_shift: int
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Read and transform data from a csv to 3 dataframe in order to perform test
    and illustrate data evaluation process
    :param path: where are the data
    :param n: size of batches
    :param data_shift: value of the shift for the third data_batch
    :return: three data_frames, the first one is the reference, the second one
    should be correct, the third one has major change in schema and
    distribution
    """
    df_batch_1 = pd.read_csv(path, skiprows=lambda x: x not in range(0, n))
    keep_rows_batch_2 = [x for x in range(n, 2 * n)] + [0]
    df_batch_2 = pd.read_csv(
        path, skiprows=lambda x: x not in keep_rows_batch_2
    )

    # Introduce NA in data
    df_batch_3 = df_batch_2.copy()
    df_batch_3.loc[0:5, "meantemp"] = None

    # Introduce a drift in data
    df_batch_4 = df_batch_2.copy()
    df_batch_4["meantemp"] = df_batch_4["meantemp"] + data_shift

    # Introduce error in schema
    df_batch_5 = df_batch_2[["meantemp", "humidity"]].copy()

    return df_batch_1, df_batch_2, df_batch_3, df_batch_4, df_batch_5


if __name__ == "__main__":
    batch_1, batch_2, batch_na, batch_shift, batch_schema = get_fake_data(
        path=DATA_PATH, n=400, data_shift=20
    )

    # Make a reference
    stats_batch_1 = tfdv.generate_statistics_from_dataframe(dataframe=batch_1)
    schema_batch_1 = tfdv.infer_schema(stats_batch_1)

    # Checking schema missing cols
    try:
        check_data_schema_anomalies_report(
            data=batch_schema, schema=schema_batch_1, severity_check=["ERROR"]
        )
    except Exception as e:
        print(e)

    # Checking schema na
    try:
        check_data_schema_anomalies_report(
            data=batch_na, schema=schema_batch_1, severity_check=["ERROR"]
        )
    except Exception as e:
        print(e)

    # Checking skew
    try:
        check_data_skew_anomalies_report(
            data=batch_shift,
            ref_schema=schema_batch_1,
            ref_stats=stats_batch_1,
            severity_check=["ERROR"],
            threshold=0.002,
            skew_thresholds={"date": 0.002},
        )
    except Exception as e:
        print(e)
