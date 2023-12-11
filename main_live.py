import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return img

def main():
    # Face Analysis Application #
    st.title("Real Time Face Application")
    st.header("Webcam Live Feed")
    
    st.write("Click on start to use webcam and detect your face emotion")
    
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    
if __name__ == "__main__":
    main()
