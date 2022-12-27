import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from config import (
    DETECTION_INFO_FOLDER,
    IMAGES_WITH_DETECTION_FOLDER,
    METADATA_INFO_FOLDER,
    OBJECT_DETECTED_FOLDER,
)
from src.predict.utils_handle_images import (
    add_detection_info_to_image,
    get_all_untreated_images_list,
    get_metadata_and_image_as_numpy_array,
)
from src.predict.utils_load_model import (
    ALL_MODELS,
    category_index,
    model_display_name,
)

tf.get_logger().setLevel("ERROR")
model_handle = ALL_MODELS[model_display_name]


def save_image_with_detection_information(image, input_image_path):
    plt.figure(figsize=(48, 64))
    plt.imshow(image)
    plt.savefig(
        os.path.join(
            IMAGES_WITH_DETECTION_FOLDER, input_image_path.split("/")[-1]
        )
    )


def save_detection_information(information, image_name):
    np.save(
        os.path.join(
            DETECTION_INFO_FOLDER,
            image_name,
        ),
        information,
    )


def save_metadata(metadata, image_name):
    metadata_json = json.dumps(metadata)
    metadata_json_file = os.path.join(
        METADATA_INFO_FOLDER,
        image_name + ".json",
    )

    with open(metadata_json_file, "w") as f:
        f.write(metadata_json)


def save_objects_detected_in_image(
    detection_information, image_np, image_name
):
    score_indices = detection_information["detection_scores"] > 0.6
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

    for i in range(boxes_int.shape[0]):
        image_focus = image_np[
            :,
            boxes_int[i][0] : boxes_int[i][2],
            boxes_int[i][1] : boxes_int[i][3],
            :,
        ][0]
        plt.imshow(image_focus)
        plt.savefig(
            os.path.join(
                image_objects_folder_path, f"object_{i}_{labels_names[i]}.png"
            )
        )


if __name__ == "__main__":

    import time

    start_load = time.perf_counter()
    print("loading model...")
    detection_model = hub.load(model_handle)
    print("model loaded!")
    end_load = time.perf_counter()

    inference_params = {
        "Selected model": model_display_name,
        "Model Handle at TensorFlow Hub": model_handle,
        "Model loading time": end_load - start_load,
    }

    images_to_treat = get_all_untreated_images_list()

    number_of_images = 0
    start_inference = time.perf_counter()
    for image_path in images_to_treat:
        image_name = image_path.split("/")[-1].split(".")[0]

        (image_np, image_metadata) = get_metadata_and_image_as_numpy_array(
            path=image_path
        )

        detection_information = detection_model(image_np)

        (
            image_with_detection_information,
            detection_information,
        ) = add_detection_info_to_image(
            image_np=image_np, detection_results=detection_information
        )

        save_image_with_detection_information(
            image=image_with_detection_information[0],
            input_image_path=image_path,
        )

        save_detection_information(
            information=detection_information, image_name=image_name
        )

        save_metadata(metadata=image_metadata, image_name=image_name)

        save_objects_detected_in_image(
            detection_information=detection_information,
            image_np=image_np,
            image_name=image_name,
        )

        number_of_images += 1

    end_inference = time.perf_counter()
    inference_params["Inference time"] = end_inference - start_inference
    inference_params["Number of images treated"] = number_of_images
