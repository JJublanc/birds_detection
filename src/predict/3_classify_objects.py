import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from keras.preprocessing import image
from tensorflow.keras import layers

from src.utils.images.utils_load_and_save import load_image_as_np_array


if __name__=="__main__":
	URL = 'https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1'               # Import pre-trained bird classification model from Tensorflow Hub
	#bird_model = hub.KerasLayer(URL, input_shape=(IMAGE_RES,IMAGE_RES,3))                 # Using aiy/vision/classifier/birds_V1 classifying 964 bird species from images. It is based on MobileNet, and trained on photos contributed by the iNaturalist community
	#bird_model.trainable=False

	bird_model = hub.load(URL)

	image_path = "/Users/johanjublanc/DataScienceProjects/birds_project/data/5_object_detected/test_oiseau/object_0_bird.png"
	load_image_as_np_array(image_path)
