import os

import numpy as np
from PIL import Image, ImageChops

from config import (
    DETECTION_INFO_FOLDER,
    IMAGES_WITH_DETECTION_FOLDER,
    INPUT_IMAGES_FOLDER,
    MAIN_FOLDER,
    PREPARED_IMAGES,
)
from src.utils.images.utils_handle_images import extract_metadata
from src.utils.images.utils_load_and_save import load_image
from src.utils.time.convert_time import get_date_from_string
from utils.images.utils_load_model import category_index

# TODO : clean the way path are handled


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        return im


def get_detections_outputs(image_name, image_name_with_extention):
    detection_information = np.load(
        os.path.join(DETECTION_INFO_FOLDER, image_name + ".npy"),
        allow_pickle=True,
    )

    image = load_image(
        os.path.join(INPUT_IMAGES_FOLDER, image_name_with_extention)
    )
    image_metadata = extract_metadata(image)

    augmented_image = Image.open(
        os.path.join(IMAGES_WITH_DETECTION_FOLDER, image_name_with_extention)
    )

    return detection_information, image_metadata, augmented_image


def get_event_info(
    detection_information, image_metadata, image_name, detection_threshold=0.6
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


def process_and_save_image(image_with_extention, image_with_detection_boxes):
    image_processed_path = os.path.join(PREPARED_IMAGES, image_with_extention)
    image_processed = image_crop_framework_and_resize(
        image_with_detection_boxes
    )
    image_processed.save(image_processed_path)

    return image_processed_path


def extract_images_info():
    image_with_extention_list = [
        item for item in os.listdir(IMAGES_WITH_DETECTION_FOLDER)
    ]
    image_list = [item.split(".")[0] for item in image_with_extention_list]
    images_info = {}

    for i, image_name in enumerate(image_list):
        if image_name != "":
            (
                detection_information,
                image_metadata,
                image_with_detection_boxes,
            ) = get_detections_outputs(
                image_name=image_name,
                image_name_with_extention=image_with_extention_list[i],
            )

            try:  # noqa: E722

                score_values, labels_names, date_info = get_event_info(
                    detection_information=detection_information,
                    image_metadata=image_metadata,
                    image_name=image_name,
                    detection_threshold=0.6,
                )

                image_processed_path = process_and_save_image(
                    image_with_extention_list[i], image_with_detection_boxes
                )

                image_info = {
                    "id": image_name,
                    "labels_names": list(labels_names),
                    "score_values": list(score_values),
                    "image_processed_path": image_processed_path,
                }

                image_info.update(date_info)
                images_info[image_name] = image_info

            except:  # noqa: E722
                pass
    return images_info


if __name__ == "__main__":
    images_info = extract_images_info()
    np.save(os.path.join(MAIN_FOLDER, "images_info"), images_info)
