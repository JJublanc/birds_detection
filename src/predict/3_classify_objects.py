

URL = 'https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1'               # Import pre-trained bird classification model from Tensorflow Hub
bird = hub.KerasLayer(URL, input_shape=(IMAGE_RES,IMAGE_RES,3))                 # Using aiy/vision/classifier/birds_V1 classifying 964 bird species from images. It is based on MobileNet, and trained on photos contributed by the iNaturalist community
bird.trainable=False