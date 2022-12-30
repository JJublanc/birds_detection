import json

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_timeline import st_timeline
from config import IMAGE_INFO_FILE
import streamlit.components.v1 as components


# ref Carousel
# https://github.com/DenizD/Streamlit-Image-Carousel
def get_time_event(object_detection_info):
	return [{"id": item["id"],
	         "content": item["content"],
	         "start": item["start"]} for item in
	        object_detection_info.values()]


if __name__ == "__main__":
	object_detection_info = np.load(IMAGE_INFO_FILE + ".npy",
	                                allow_pickle=True).item()
	time_events_list = get_time_event(object_detection_info)
	st.set_page_config(layout="wide")

	timeline = st_timeline(
		time_events_list,
		groups=[],
		options={},
		height="300px"
	)

	st.subheader("Selected item")
	if timeline is not None:
		date_time = timeline["start"]
		text = f"{date_time} \n \n"
		x = " "

		images_list = [Image.open(
			object_detection_info[timeline["id"]]["image_processed_path"])]

		caption_list = [timeline["id"]]
		print(object_detection_info[timeline["id"]])
		for item in object_detection_info[timeline["id"]]:
			if item.split("_")[0] == "object":
				image = Image.open(
					object_detection_info[timeline["id"]]
					[item]
					["object_path"]
				)
				print(object_detection_info[timeline["id"]]
					[item]
					["object_path"])
				bird_type = object_detection_info[timeline["id"]][item]["label"]
				proba = object_detection_info[timeline["id"]][item]["proba"]
				print(image.size)
				image = image.resize(
					(200, int(200 * (image.size[1] / image.size[0]))))
				print(image.size)
				images_list.append(image)

				caption = f"type d'oiseau : {bird_type} \n proba : {int(proba * 100)}%"
				caption_list.append(caption)

		st.image(images_list, width=200, caption=caption_list)
