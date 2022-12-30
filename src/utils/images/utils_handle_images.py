import os
from typing import List
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from PIL.ExifTags import TAGS

from config import IMAGES_WITH_DETECTION_FOLDER, INPUT_IMAGES_FOLDER
from utils.images.utils_load_model import (
    COCO17_HUMAN_POSE_KEYPOINTS,
    category_index,
)


def extract_metadata(image):
    # extracting the exif metadata
    exifdata = image.getexif()

    metadata = {}
    # looping through all the tags present in exifdata
    for tagid in exifdata:
        # getting the tag name instead of tag id
        tagname = TAGS.get(tagid, tagid)

        # passing the tagid to get its respective value
        value = str(exifdata.get(tagid))

        # printing the final result
        metadata[tagname] = value

    return metadata


def convert_image_to_np_array(image):
    (im_width, im_height) = image.size
    return (
        np.array(image.getdata())
        .reshape((1, im_height, im_width, 3))
        .astype(np.uint8)
    )


def get_all_untreated_images_list() -> List:
    input_images_list = [file for file in os.listdir(INPUT_IMAGES_FOLDER)]
    images_with_detection_list = [
        file for file in os.listdir(IMAGES_WITH_DETECTION_FOLDER)
    ]

    all_untreated_images_path_list = []
    for file in input_images_list:
        if file not in images_with_detection_list:
            all_untreated_images_path_list.append(
                os.path.join(INPUT_IMAGES_FOLDER, file)
            )

    return all_untreated_images_path_list


def transform_image(
    image_np, flip_image_horizontally, convert_image_to_grayscale
):
    # Flip horizontally
    if flip_image_horizontally:
        image_np[0] = np.fliplr(image_np[0]).copy()

    # Convert image to grayscale
    if convert_image_to_grayscale:
        image_np[0] = np.tile(
            np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)
        ).astype(np.uint8)


def add_detection_info_to_image(image_np, detection_results):
    result = {key: value.numpy() for key, value in detection_results.items()}

    label_id_offset = 0
    image_np_with_detections = image_np.copy()

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if "detection_keypoints" in result:
        keypoints = result["detection_keypoints"][0]
        keypoint_scores = result["detection_keypoint_scores"][0]

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections[0],
        result["detection_boxes"][0],
        (result["detection_classes"][0] + label_id_offset).astype(int),
        result["detection_scores"][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS,
    )

    return image_np_with_detections, result
