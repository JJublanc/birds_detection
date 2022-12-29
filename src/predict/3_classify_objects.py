import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from config import OBJECT_DETECTED_FOLDER
from src.utils.images.utils_load_and_save import load_image_as_np_array

IMAGE_RES = 224
LABEL = "nom"


def get_birds_model():
    URL = "https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1"
    bird_model = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
    bird_model.trainable = False
    tf_model = tf.keras.Sequential([bird_model])
    return tf_model


if __name__ == "__main__":
    model = get_birds_model()
    labels = pd.read_csv(
        "./data/labels_oiseaux.csv", sep=";", header=0, index_col=0
    )  # file providing species in french, english and latin

    for path, subdirs, files in os.walk(OBJECT_DETECTED_FOLDER):
        for name in files:
            if name.split(".")[0].split("_")[-1] == "bird":
                image_path = os.path.join(path, name)
                image_np = load_image_as_np_array(
                    image_path, new_size=(IMAGE_RES, IMAGE_RES)
                )

                output = model.predict(image_np / 255.0)  # get prediction
                prediction = np.argmax(tf.squeeze(output).numpy())
                print(np.max(tf.squeeze(output).numpy()))
                print((labels[LABEL][prediction]))
