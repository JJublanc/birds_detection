import datetime
import json
import os

import numpy as np
from PIL import Image, ImageChops

from config import (
	DETECTION_INFO_FOLDER,
	IMAGES_WITH_DETECTION_FOLDER,
	METADATA_INFO_FOLDER,
	PREPARED_IMAGES,
)
from utils.images.utils_load_model import category_index


def save_metadata(metadata, image_name):
	metadata_json = json.dumps(metadata)
	metadata_json_file = os.path.join(
		METADATA_INFO_FOLDER,
		image_name + ".json",
	)

	with open(metadata_json_file, "w") as f:
		f.write(metadata_json)


def trim(im):
	bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
	diff = ImageChops.difference(im, bg)
	diff = ImageChops.add(diff, diff, 2.0, -100)
	bbox = diff.getbbox()
	if bbox:
		return im.crop(bbox)
	else:
		return im


def get_date_from_string(s_date):
	date_patterns = [
		"%Y-%m-%d %H:%M:%S",
		"%Y:%m:%d %H:%M:%S",
		"%Y-%m-%d",
		"%d-%m-%Y",
	]

	for pattern in date_patterns:
		try:
			return datetime.datetime.strptime(s_date, pattern)
		except ValueError:
			print("Date is not in expected format: %s" % (s_date))


def get_detections_outputs(image_name, image_name_with_extention):
	detection_information = np.load(
		os.path.join(DETECTION_INFO_FOLDER, image_name + ".npy"),
		allow_pickle=True,
	)

	with open(
			os.path.join(METADATA_INFO_FOLDER, image_name + ".json"), "r"
	) as f:
		image_metadata = json.load(f)

	augmented_image = Image.open(
		os.path.join(IMAGES_WITH_DETECTION_FOLDER, image_name_with_extention)
	)

	return detection_information, image_metadata, augmented_image


def get_event_info(
		detection_information, image_metadata, image_name,
		detection_threshold=0.6
):
	score_indices = np.where(
		detection_information.item()["detection_scores"] > detection_threshold
	)[1]

	score_values = detection_information.item()["detection_scores"][0][
		score_indices
	]

	labels_values = detection_information.item()["detection_classes"][0][
		score_indices
	]

	labels_names = [category_index[i]["name"] for i in labels_values]

	date_formated = get_date_from_string(image_metadata["DateTime"]).strftime(
		"%Y-%m-%d %H:%M:%S"
	)

	date_item = {
		"id": image_name.split(".")[0],
		"content": date_formated,
		"start": date_formated,
	}

	return score_values, labels_names, date_item


def image_crop_framework_and_resize(image, newsize=(300, 200)):
	augmented_image_cropped = trim(image)
	return augmented_image_cropped.resize(newsize)


def get_image_information_and_transform_and_save_images():
	images_information = {}
	image_with_extention_list = [
		item for item in os.listdir(IMAGES_WITH_DETECTION_FOLDER)
	]
	image_list = [item.split(".")[0] for item in image_with_extention_list]

	time_events_list = []
	object_detection_info = {}

	for i, image_name in enumerate(image_list):

		(
			detection_information,
			image_metadata,
			image_with_detection_boxes,
		) = get_detections_outputs(
			image_name=image_name,
			image_name_with_extention=image_with_extention_list[i],
		)

		try:  # noqa: E722

			score_values, labels_names, date_item = get_event_info(
				detection_information=detection_information,
				image_metadata=image_metadata,
				image_name=image_name,
				detection_threshold=0.6,
			)

			images_information[image_name] = image_metadata

			time_events_list.append(date_item)

			image_processed_path = os.path.join(
				PREPARED_IMAGES, image_with_extention_list[i]
			)
			image_processed = image_crop_framework_and_resize(
				image_with_detection_boxes
			)
			image_processed.save(image_processed_path)

			object_detection_info[image_name] = {
				"labels_names": list(labels_names),
				"scorer_values": list(score_values),
				"image_path": image_processed_path,
			}
		except:  # noqa: E722
			pass
	return object_detection_info, time_events_list


if __name__ == "__main__":
	# metadata = extract_metadata(image)

	(
		object_detection_info,
		time_events_list,
	) = get_image_information_and_transform_and_save_images()

	np.save("app/object_detection_info", object_detection_info)

	with open("app/time_events_list", "w") as f:
		json.dump(time_events_list, f)
