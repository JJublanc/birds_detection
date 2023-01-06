import cv2
import os

DATA_PATH = "data_video"
VIDEO_NAME = 'video/birds_video.mp4'

def extract():
  vidcap = cv2.VideoCapture(os.path.join(DATA_PATH, VIDEO_NAME))
  success, image = vidcap.read()
  count = 0
  while success:
    cv2.imwrite(os.path.join(DATA_PATH, "1_input_images/frame%d.jpg" % count), image)
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 50
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)

if __name__=="__main__":
  extract()