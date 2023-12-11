import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        frame_rgb_after = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #rgb_after = Image.fromarray(frame_rgb_after)
        font = cv2.FONT_HERSHEY_SIMPLEX
        input_text_instruction = "--- SIDE ---"
        cv2.putText(frame_rgb_after,input_text_instruction,(int(width/4), int(height/2)), font,1,(255,255,255),2)
        img = cv2.cvtColor(frame_rgb_after, cv2.COLOR_BGR2RGB)
        return img

def main():
    # Face Analysis Application #
    st.title("Real Time sizing Application")
    
    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect your face emotion")
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)



if __name__ == "__main__":
    main()
