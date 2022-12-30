import os

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from config import OBJECT_DETECTED_FOLDER
from src.utils.images.utils_load_and_save import load_image_as_np_array
from src.utils.ml_versioning_wrapper.synch_wrapper import tracking_wrapper

IMAGE_RES = 224
LABEL = "nom"
import numpy as np
from src.utils.time.wrapper_timer import timer_wrapper
from config import IMAGE_INFO_FILE

# ref:
# https://github.com/LaurentVeyssier/Bird_Classifier_Tensorflow_Colab_Notebook

def get_birds_model():
	URL = "https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1"
	bird_model = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
	bird_model.trainable = False
	tf_model = tf.keras.Sequential([bird_model])
	return tf_model


def classify_object(object_folder,
                    object_file,
                    model,
                    labels):
	image_objects_info = {}
	object_image_path = os.path.join(object_folder, object_file)
	image_np = load_image_as_np_array(
		object_image_path, new_size=(IMAGE_RES, IMAGE_RES)
	)
	output = model.predict(image_np / 255.0)  # get prediction
	prediction = np.argmax(tf.squeeze(output).numpy())

	image_objects_info["object_type"] = \
		object_file.split(".")[0].split("_")[-1]
	image_objects_info["proba"] = np.max(
		tf.squeeze(output).numpy())
	image_objects_info["label"] = labels[LABEL][prediction]
	image_objects_info["object_path"] = object_image_path

	return image_objects_info


@timer_wrapper
def main_classify_birds():
	model = get_birds_model()
	labels = pd.read_csv(
		"./data/labels_oiseaux.csv", sep=";", header=0, index_col=0
	)  # file providing species in french, english and latin

	images_info_dict = np.load(IMAGE_INFO_FILE + ".npy",
	                           allow_pickle=True).item()
	for image_name in images_info_dict.keys():
		for object_folder, subdirs, files in os.walk(
				os.path.join(OBJECT_DETECTED_FOLDER,
				             image_name)):
			for object_file in files:
				image_objects_info = classify_object(object_folder,
				                                     object_file,
				                                     model,
				                                     labels)

				images_info_dict[image_name][
					object_file.split(".")[0]] = image_objects_info
	return images_info_dict


@tracking_wrapper
def main():
	images_info_dict, inference_params = main_classify_birds(timer_key="Inference time")
	mlflow.log_params(inference_params)
	return images_info_dict


if __name__ == "__main__":
	images_info_dict = main(experiment_branch="main",
	                        wrapper_experiment_name=
	                        "birds_classification"
	                        )

	np.save(IMAGE_INFO_FILE, images_info_dict)
