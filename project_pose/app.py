import threading

import cv2
import numpy as np
import requests
import streamlit as st

from pose import poseDetector
from tools import generer_description

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except ImportError:
    av = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False


st.title("IA de detection de posture")


@st.cache_resource
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Impossible de charger le classifieur visage: {cascade_path}")
    return cascade


def draw_faces(image_bgr, face_cascade):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            image_bgr,
            "Visage",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    return image_bgr, len(faces)


def build_pose_data(lm_list):
    if not lm_list:
        return "Aucune donnee de pose disponible."
    return "\n".join([f"ID {p[0]}: x={p[1]}, y={p[2]}" for p in lm_list])


def call_bob(user_question, pose_data):
    payload = {
        "user_question": user_question,
        "pose_data": pose_data,
    }

    try:
        response = requests.post("http://127.0.0.1:8000/analyze", json=payload, timeout=25)
        response.raise_for_status()
        return response.json().get("response", "Désolé, je n'ai pas pu analyser.")
    except Exception as e:
        return f"Impossible de contacter BOB. Verifie que agent.py tourne. (Erreur: {e})"


class PoseFaceProcessor(VideoProcessorBase):
    def __init__(self, face_cascade):
        self.detector = poseDetector(mode=False)
        self.face_cascade = face_cascade
        self.lock = threading.Lock()
        self.last_lm_list = []

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        annotated = self.detector.findPose(image, draw=True)
        lm_list = self.detector.findPosition(annotated, draw=False)
        annotated, face_count = draw_faces(annotated, self.face_cascade)

        status = "Pose detectee" if lm_list else "Aucune pose"
        cv2.putText(
            annotated,
            f"{status} | Visages: {face_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        with self.lock:
            self.last_lm_list = lm_list

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_pose_data" not in st.session_state:
    st.session_state.last_pose_data = "Aucune donnee de pose disponible."
if "last_image_id" not in st.session_state:
    st.session_state.last_image_id = None

with st.sidebar:
    mode = st.radio("Source", ["Image", "Camera"])

if mode == "Image":
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        detector = poseDetector(mode=True)
        face_cascade = load_face_cascade()

        file_content = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(file_content), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_id = f"{uploaded_file.name}:{len(file_content)}"

        if opencv_image is None:
            st.error("Image invalide ou non lisible.")
        else:
            annotated = detector.findPose(opencv_image.copy())
            lm_list = detector.findPosition(annotated, draw=False)
            annotated, face_count = draw_faces(annotated, face_cascade)

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Resultat")
            st.info(generer_description(detector.results))
            st.caption(f"Visages detectes: {face_count}")

            st.session_state.last_pose_data = build_pose_data(lm_list)

            if lm_list and st.session_state.last_image_id != image_id:
                with st.spinner("BOB analyse votre posture..."):
                    reponse_bob = call_bob(
                        "Fais une analyse physique et biomecanique complete de cette posture a partir des points fournis.",
                        st.session_state.last_pose_data,
                    )
                    st.session_state.messages.append({"role": "assistant", "content": reponse_bob})
                    st.session_state.last_image_id = image_id

            if detector.results and detector.results.pose_landmarks:
                st.success("Posture detectee.")
            else:
                st.warning("Aucune posture detectee.")

else:
    st.subheader("Camera en direct")

    if not WEBRTC_AVAILABLE:
        st.warning(
            "Le mode camera temps reel nécessite streamlit-webrtc. "
            "Installe-le avec: pip install streamlit-webrtc"
        )
    else:
        face_cascade = load_face_cascade()

        ctx = webrtc_streamer(
            key="pose-face-live",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: PoseFaceProcessor(face_cascade),
            async_processing=True,
        )

        if ctx and ctx.video_processor:
            processor = ctx.video_processor
            with processor.lock:
                current_lm_list = list(processor.last_lm_list)
            if current_lm_list:
                st.session_state.last_pose_data = build_pose_data(current_lm_list)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("En quoi puis-je t'aider ?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("BOB reflechit..."):
            reponse = call_bob(prompt, st.session_state.last_pose_data)
            st.markdown(reponse)
            st.session_state.messages.append({"role": "assistant", "content": reponse})
