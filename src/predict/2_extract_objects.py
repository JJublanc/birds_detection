import os
import mlflow
import numpy as np

from config import (
	DETECTION_INFO_FOLDER,
	IMAGES_WITH_DETECTION_FOLDER,
	INPUT_IMAGES_FOLDER,
	OBJECT_DETECTED_FOLDER,
	OBJECT_DETECTION_THRESHOLD,
	MARGIN_X,
	MARGIN_Y
)

from src.utils.images.utils_load_and_save import (
	get_detections_outputs,
	save_np_array_as_jpeg
)

from src.utils.ml_versioning_wrapper.synch_wrapper import tracking_wrapper
from src.utils.time.wrapper_timer import timer_wrapper
from utils.images.utils_load_and_save import load_image_as_np_array
from utils.images.utils_load_model import category_index


def save_objects_detected_in_image(
		detection_information, image_np, image_name
):
	score_indices = detection_information["detection_scores"] > \
	                OBJECT_DETECTION_THRESHOLD
	boxes = detection_information["detection_boxes"][:, score_indices[0], :]
	scale = np.concatenate(
		(image_np.shape[1:3], image_np.shape[1:3]), axis=None
	)
	boxes_int = (boxes * scale).astype(int)[0]
	image_objects_folder_path = os.path.join(
		OBJECT_DETECTED_FOLDER, image_name
	)
	# TODO : mutualize code with prepare_data in app
	# TODO : bring back this code in predict
	os.makedirs(image_objects_folder_path, exist_ok=True)
	labels_values = detection_information["detection_classes"][0][
		score_indices[0]
	]
	labels_names = [category_index[i]["name"] for i in labels_values]
	size = image_np.shape
	for i in range(boxes_int.shape[0]):
		image_focus = image_np[
		              :,
		              max(boxes_int[i][0] - MARGIN_Y, 0):
		              min(boxes_int[i][2] + MARGIN_Y, size[1]),
		              max(boxes_int[i][1] - MARGIN_X, 0):
		              min(boxes_int[i][3] + MARGIN_X, size[2]),
		              :,
		              ][0]

		save_np_array_as_jpeg(image_np=image_focus,
		                      path=os.path.join(
			                      image_objects_folder_path,
			                      f"object_{i}_{labels_names[i]}.jpeg"
		                      ))


def get_images_list(full_list, to_exclude_list):
	result_list = []
	to_exclude_lits = [item.split(".")[0] for item in to_exclude_list]
	for item in full_list:
		if item.split(".")[0] not in to_exclude_lits:
			result_list.append(item)
	return result_list


@timer_wrapper
def exatract_objects():
	number_of_images = 0
	images_which_objects_are_extracted = os.listdir(OBJECT_DETECTED_FOLDER)
	images_list = os.listdir(INPUT_IMAGES_FOLDER)
	images_to_handle_list = get_images_list(
		images_list, images_which_objects_are_extracted
	)

	for image_name_with_extention in images_to_handle_list:
		image_name = image_name_with_extention.split(".")[0]
		image_path = os.path.join(
			INPUT_IMAGES_FOLDER, image_name_with_extention
		)
		image_np = load_image_as_np_array(image_path)

		detection_information, augmented_image = get_detections_outputs(
			image_name=image_name,
			image_name_with_extention=image_name_with_extention,
			detection_info_folder=DETECTION_INFO_FOLDER,
			images_with_detection_folder=IMAGES_WITH_DETECTION_FOLDER,
		)

		save_objects_detected_in_image(
			detection_information=detection_information,
			image_np=image_np,
			image_name=image_name,
		)

		number_of_images += 1

	return number_of_images


@tracking_wrapper
def main_extract_objects():
	timer_key = "Extraction elapsed time"
	number_of_images, object_extraction_params = exatract_objects(
		timer_key=timer_key
	)

	object_extraction_params["Number of images treated"] = number_of_images

	if number_of_images > 0:
		object_extraction_params["Extraction time per image"] = (
				object_extraction_params[timer_key] / number_of_images
		)

	mlflow.log_params(object_extraction_params)


if __name__ == "__main__":
	main_extract_objects(
		experiment_branch="main", wrapper_experiment_name="objects_extraction"
	)
