import os
import sys

import mlflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from src.utils.ml_versioning_wrapper.synch_wrapper import tracking_wrapper
from src.train.training_config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, \
    NUM_EPOCH

IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_SHAPE = IMG_SIZE + (3,)


def get_model(num_classes, image_batch):
    ##############
    # preprocess #
    ##############

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    ##############
    # Base model #
    ##############

    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    feature_batch = base_model(image_batch)

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average)

    prediction_layer = tf.keras.layers.Dense(num_classes)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch)

    #####################
    # Data augmentation #
    #####################

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(IMG_HEIGHT,
                                           IMG_WIDTH,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.summary()

    return model


@tracking_wrapper
def train_model(model, train_ds, test_ds, epochs):
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs
    )


def main():
    data_path = sys.argv[1]
    uri = sys.argv[2]
    name = sys.argv[3]

    # TODO : add the preprocess data file
    # Data
    train_ds, val_ds = preprocess_data(data_path,
                                       IMG_WIDTH,
                                       IMG_HEIGHT,
                                       BATCH_SIZE)
    class_names = train_ds.class_names
    num_classes = len(class_names)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    image_batch, labels_batch = next(iter(train_ds))
    print("image_batch : ", image_batch, "labels_batch : ", labels_batch)

    # Model
    model = get_model(num_classes, image_batch)

    # Train
    mlflow.set_tracking_uri(uri)
    mlflow.autolog()
    mlflow.set_tag("USER", name)
    train_model(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        epochs=NUM_EPOCH)


if __name__ == "__main__":
    main()
