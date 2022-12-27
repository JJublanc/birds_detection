import pandas as pd
from pandas_profiling import ProfileReport


def create_data_profile_from_csv(
    path_origin: str, destination_file_name: str, nb_of_sample=100
) -> None:
    """
    create a data profile with pandas_profiling
    :param path_origin: where are the data
    :param destination_file_name: name of the output html file
    :param nb_of_sample: number of line sample in the file
    :return: None
    """
    df = pd.read_csv(path_origin)
    profile = ProfileReport(df.sample(n=nb_of_sample), title="Report")
    profile.to_file(f"{destination_file_name}.html")
