import os
MAIN_FOLDER = "./data_video_2"
INPUT_IMAGES_FOLDER = os.path.join(MAIN_FOLDER, "1_input_images")
IMAGES_WITH_DETECTION_FOLDER = os.path.join(MAIN_FOLDER, "2_images_with_detection_boxes")
DETECTION_INFO_FOLDER = os.path.join(MAIN_FOLDER, "3_detection_info")
METADATA_INFO_FOLDER = os.path.join(MAIN_FOLDER, "4_images_metadata")
PREPARED_IMAGES = os.path.join(MAIN_FOLDER, "6_prepared_images_for_dashboard")
OBJECT_DETECTED_FOLDER = os.path.join(MAIN_FOLDER, "5_object_detected")
TRAIN_DATA_FOLDER = os.path.join(MAIN_FOLDER, "7_training_birds_dataset")
PREPROCESS_DATA_FOLDER = os.path.join(MAIN_FOLDER, "7_training_birds_dataset/preprocessed_images")
TRACKING_URI = "./"
IMAGE_INFO_FILE = os.path.join(MAIN_FOLDER, "images_info")
ANALYSIS_FOLDER = "./results/analysis"
OBJECT_DETECTION_THRESHOLD = 0.2
MARGIN_X = 20
MARGIN_Y = 20

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
BATCH_SIZE = 32

DATA_PATH = "data_video_2"
VIDEO_PATH = os.path.join(MAIN_FOLDER, "video/birds_and_cats_long_video.mp4")

