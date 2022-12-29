import numpy as np
import tensorflow as tf
import os
from PIL import Image

from six import BytesIO
from six.moves.urllib.request import urlopen

from src.utils.images.utils_handle_images import convert_image_to_np_array


def get_detections_outputs(image_name,
                           image_name_with_extention,
                           detection_info_folder,
                           images_with_detection_folder
                           ):
	detection_information = np.load(
		os.path.join(detection_info_folder, image_name + ".npy"),
		allow_pickle=True,
	)

	augmented_image = Image.open(
		os.path.join(images_with_detection_folder, image_name_with_extention)
	)

	return detection_information.item(), augmented_image


def load_image(path):
	if path.startswith("http"):
		response = urlopen(path)
		image_data = response.read()
		image_data = BytesIO(image_data)
		image = Image.open(image_data)
	else:
		image_data = tf.io.gfile.GFile(path, "rb").read()
		image = Image.open(BytesIO(image_data))
	return image


def load_image_as_np_array(image_path):
	image = load_image(path=image_path)
	return convert_image_to_np_array(image)


def load_image_and_detect_objects(image_path, detection_model):
	image_name = image_path.split("/")[-1].split(".")[0]
	image_np = load_image_as_np_array(image_path)
	detection_information = detection_model(image_np)

	return image_np, detection_information, image_name
