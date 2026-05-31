import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_drawing_module
import numpy as np
import pickle
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import LSTM
import av

# Configuration
SEQUENCE_LENGTH = 10


# Keras 3 removed the `time_major` argument from LSTM.
# This shim lets us load models saved with Keras 2 (.h5).
class _CompatLSTM(LSTM):
    def __init__(self, *args, time_major=False, **kwargs):
        super().__init__(*args, **kwargs)


@st.cache_resource
def load_ai_model():
    model = tf.keras.models.load_model(
        "weights/mha_gesturenet.h5",
        custom_objects={"LSTM": _CompatLSTM},
        compile=False,
    )
    with open("weights/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder


class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.model, self.label_encoder = load_ai_model()
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.hands = mp_hands_module.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        predicted_label = "No Gesture"
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing_module.draw_landmarks(
                    img, hand_landmarks, mp_hands_module.HAND_CONNECTIONS
                )
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                if len(landmarks) == 42:
                    self.sequence_buffer.append(landmarks)

                if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                    input_seq = np.expand_dims(
                        np.array(self.sequence_buffer, dtype=np.float32), axis=0
                    )
                    prediction = self.model.predict(input_seq, verbose=0)
                    predicted_class = np.argmax(prediction)
                    confidence = float(np.max(prediction))
                    try:
                        predicted_label = self.label_encoder.inverse_transform(
                            [predicted_class]
                        )[0]
                    except Exception:
                        predicted_label = str(predicted_class)

        text = f"{predicted_label} {confidence:.2f}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(page_title="MHA · GestureNet (Live)", layout="wide")
    st.title("MHA-GestureNet — Live (WebRTC)")
    st.write("Allow webcam access when prompted, then click START.")

    webrtc_streamer(
        key="gesture",
        video_processor_factory=GestureProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )


if __name__ == "__main__":
    main()
