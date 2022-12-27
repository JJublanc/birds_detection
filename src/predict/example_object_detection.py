import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from src.predict.utils_handle_images import (
    add_results_to_image_and_save,
    load_all_untreated_images,
    load_image_into_numpy_array,
)
from src.predict.utils_load_model import ALL_MODELS, model_display_name

tf.get_logger().setLevel("ERROR")
model_handle = ALL_MODELS[model_display_name]

if __name__ == "__main__":
    params = {}
    params["Selected model"] = model_display_name
    params["Model Handle at TensorFlow Hub"] = model_handle

    print("loading model...")
    hub_model = hub.load(model_handle)
    print("model loaded!")

    images_to_treat = load_all_untreated_images()
    for image_path in images_to_treat:
        image_np, metadata = load_image_into_numpy_array(image_path)
        results = hub_model(image_np)
        image_with_results, results = add_results_to_image_and_save(
            image_np, results
        )
        results.update(metadata)
        plt.figure(figsize=(48, 64))
        plt.imshow(image_with_results[0])
        plt.savefig(os.path.join("./data_output", image_path.split("/")[-1]))

        np.save(
            os.path.join(
                "./data_output", image_path.split("/")[-1].split(".")[0]
            ),
            results,
        )
