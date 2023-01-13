import numpy as np
from config import INPUT_IMAGES_FOLDER, ANALYSIS_FOLDER
import os
import pandas as pd
import plotly.express as px
import plotly.offline as offline


def plot_labels_freq(df, filename):
	# Assuming your dataframe is called 'df'
	df_counts = df[['label', 'label_proba']].groupby('label').count()
	df_counts['frequency'] = df_counts['label_proba'] / df.shape[0]
	fig = px.bar(df_counts.reset_index(), x='label', y='frequency')
	offline.plot(fig, filename=f'{filename}.html')


def create_dataframe():
	info = np.load("./data_video/images_info.npy", allow_pickle=True).item()
	data_to_analyse = pd.DataFrame(columns=['frame_name',
	                                        'label',
	                                        'label_proba',
	                                        'object_type'])
	images_with_objects = info.keys()

	for image_file in os.listdir(INPUT_IMAGES_FOLDER):
		image_info = {}
		image_name = image_file.split(".")[0]
		image_info["frame_name"] = image_name
		if image_name in images_with_objects:
			for key in info[image_name].keys():
				if key.split("_")[0]=="object":
					image_info["label"] = info[image_name][key]["label"]
					image_info["label_proba"] = info[image_name][key]["proba"]
					image_info["object_type"] = info[image_name][key]["object_type"]
				else:
					pass
		else:
			image_info["label"]=None
			image_info["label_proba"]=None
			image_info["object_type"]=None
		data_to_analyse = data_to_analyse.append(image_info, ignore_index=True)
	return data_to_analyse


if __name__=="__main__":
	data_to_analyse = create_dataframe()
	freq_birds = data_to_analyse[(data_to_analyse["object_type"]=="bird") &
	                             (data_to_analyse["label_proba"]>0.5)]
	plot_labels_freq(freq_birds, os.path.join(ANALYSIS_FOLDER, "birds_freq"))
