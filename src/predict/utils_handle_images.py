import os

import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image
from PIL.ExifTags import TAGS
from six import BytesIO
from six.moves.urllib.request import urlopen

from src.predict.utils_load_model import (
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
        value = exifdata.get(tagid)

        # printing the final result
        metadata[tagname] = value

    return metadata


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = None
    if path.startswith("http"):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, "rb").read()
        image = Image.open(BytesIO(image_data))

    metadata = extract_metadata(image)

    (im_width, im_height) = image.size
    return (
        np.array(image.getdata())
        .reshape((1, im_height, im_width, 3))
        .astype(np.uint8),
        metadata,
    )


def load_all_untreated_images():
    path_input = "./data_input"
    path_output = "./data_output"
    picture_path_list = [file for file in os.listdir(path_input)]
    results_path_list = [file for file in os.listdir(path_output)]

    images = []
    for file in picture_path_list:
        if file not in results_path_list:
            images.append(os.path.join(path_input, file))

    return images


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


def add_results_to_image_and_save(image_np, results):
    result = {key: value.numpy() for key, value in results.items()}

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
