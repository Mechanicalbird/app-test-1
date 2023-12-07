import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class OpenCVVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        return frame.to_ndarray()

def main():
    st.title("Streamlit WebRTC + OpenCV Example")
    st.write("Live Video Stream")

    webrtc_ctx = webrtc_streamer(
        key="example-opencv",
        video_transformer_factory=OpenCVVideoTransformer,
        async_transform=True,
    )

    if webrtc_ctx.video_transformer:
        st.write("Processed Video Stream")
        st.image(webrtc_ctx.video_transformer.output_frame)

if __name__ == "__main__":
    main()
