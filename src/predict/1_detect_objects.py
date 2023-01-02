import os
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from config import DETECTION_INFO_FOLDER, IMAGES_WITH_DETECTION_FOLDER
from src.utils.ml_versioning_wrapper.synch_wrapper import tracking_wrapper
from src.utils.time.wrapper_timer import timer_wrapper
from utils.images.utils_handle_images import (
    add_detection_info_to_image,
    get_all_untreated_images_list
)
from utils.images.utils_load_and_save import (
    load_image_and_detect_objects,
    save_np_array_as_jpeg
)

from utils.images.utils_load_model import ALL_MODELS, MODEL_DISPLAY_NAME

tf.get_logger().setLevel("ERROR")
MODEL_HANDLE = ALL_MODELS[MODEL_DISPLAY_NAME]


def save_detection_information(information, image_name):
    np.save(
        os.path.join(
            DETECTION_INFO_FOLDER,
            image_name,
        ),
        information,
    )


@timer_wrapper
def load_model():
    model = hub.load(MODEL_HANDLE)
    return model


def save_detection_info_and_augmented_image(
    detection_information_dict,
    image_name,
    image_path,
    image_with_detection_information,
):
    save_np_array_as_jpeg(
        image_np=image_with_detection_information[0],
        path=image_path,
    )

    save_detection_information(
        information=detection_information_dict, image_name=image_name
    )


@tracking_wrapper
def main_detect_object():
    detection_model, inference_params = load_model(
        input_params={
            "Selected model": MODEL_DISPLAY_NAME,
            "Model Handle at TensorFlow Hub": MODEL_HANDLE,
        },
        timer_key="Model loading time",
    )

    images_to_treat = get_all_untreated_images_list()

    number_of_images = 0
    start_inference = time.perf_counter()
    for image_path in images_to_treat:
        (
            image_np,
            detection_information,
            image_name,
        ) = load_image_and_detect_objects(image_path, detection_model)

        (
            image_with_detection_information,
            detection_information_dict,
        ) = add_detection_info_to_image(
            image_np=image_np, detection_results=detection_information
        )

        save_detection_info_and_augmented_image(
            detection_information_dict=detection_information_dict,
            image_name=image_name,
            image_path=os.path.join(IMAGES_WITH_DETECTION_FOLDER,
                                    image_name + ".jpg"),
            image_with_detection_information=image_with_detection_information,
        )

        number_of_images += 1

    end_inference = time.perf_counter()
    inference_params["Inference time"] = end_inference - start_inference
    inference_params["Number of images treated"] = number_of_images
    inference_params["Inference time per image"] = (
        end_inference - start_inference
    ) / number_of_images

    mlflow.log_params(inference_params)


if __name__ == "__main__":
    main_detect_object(
        experiment_branch="main", wrapper_experiment_name="object_detection"
    )
