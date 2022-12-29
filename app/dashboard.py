import json

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_timeline import st_timeline

if __name__ == "__main__":
    object_detection_info = np.load(
        "./data/object_detection_info.npy", allow_pickle=True
    )
    with open("data/time_events_list", "r") as f:
        time_events_list = json.load(f)

    st.set_page_config(layout="wide")
    timeline = st_timeline(
        time_events_list, groups=[], options={}, height="300px"
    )

    st.subheader("Selected item")
    if timeline is not None:
        date_time = timeline["start"]
        text = f"{date_time} \n \n"
        x = " "
        for i, label in enumerate(
            object_detection_info.item()[timeline["id"]]["labels_names"]
        ):
            proba = object_detection_info.item()[timeline["id"]][
                "scorer_values"
            ][i]
            text += f"{label} : {round(proba * 100)}% - - "

        st.write(text)
        st.image(
            Image.open(
                object_detection_info.item()[timeline["id"]]["image_path"]
            )
        )
