import datetime
import os

import cv2
import piexif
from PIL import Image

from config import INPUT_IMAGES_FOLDER, MAIN_FOLDER

DATA_PATH = "data_video"
VIDEO_PATH = os.path.join(MAIN_FOLDER, "video/birds_video.mp4")


def extract(period=50):
    video = cv2.VideoCapture(VIDEO_PATH)
    success = True
    count = 0
    now = datetime.datetime.now()

    while success:
        date_time = now + datetime.timedelta(seconds=count)
        date_time_str = date_time.strftime("%Y:%m:%d %H:%M:%S")

        success, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        exif_dict = {"0th": {piexif.ImageIFD.DateTime: None}}
        exif_dict["0th"][piexif.ImageIFD.DateTime] = date_time_str
        exif_bytes = piexif.dump(exif_dict)

        # Save the image as a JPEG file with the metadata
        image.save(
            os.path.join(INPUT_IMAGES_FOLDER, f"frame_{count}.jpg"),
            "JPEG",
            exif=exif_bytes,
            optimize=True,
        )

        print("Read a new frame: ", success)
        count += period
        video.set(cv2.CAP_PROP_POS_FRAMES, count)


if __name__ == "__main__":
    extract()
