import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
from pathlib import Path
import pandas as pd

def plot_timeseries_df_to_html(
		path: Path,
		df: pd.DataFrame,
		date_column: str = "date",
):
	fig = go.Figure()

	for col in df.columns:
		if is_numeric_dtype(df[col]):
			fig.add_trace(
				go.Scatter(
					x=df[date_column],
					y=df[col],
					name=col,
				)
			)

	fig.write_html(path)
