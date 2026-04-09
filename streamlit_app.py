"""Streamlit UI for Blink Rate Eye Strain Monitor.

Run:
    streamlit run streamlit_app.py
"""
import time
from typing import Optional

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from main import (
    LEFT_EYE_IDX,
    LEFT_EAR_IDX,
    RIGHT_EYE_IDX,
    RIGHT_EAR_IDX,
    STATUS_ALERT,
    STATUS_OK,
    check_alert,
    compute_live_blink_rate_per_min,
    detect_blink,
    draw_eye_landmarks,
    get_eye_width,
    get_face_center_and_width,
    has_required_landmarks,
    mp_face_mesh,
    update_blink_rate,
    calculate_ear,
    MIN_FACE_WIDTH_PX,
    FACE_MOVE_THRESHOLD_RATIO,
    MIN_EYE_WIDTH_PX,
    MAX_EYE_WIDTH_CHANGE_RATIO,
    EAR_THRESHOLD,
    MIN_DYNAMIC_EAR_THRESHOLD,
    EAR_BASELINE_ALPHA,
    EAR_DYNAMIC_RATIO,
)


class BlinkProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.blink_state = {
            "eye_closed": False,
            "closed_frame_count": 0,
            "last_blink_time": 0.0,
            "ear_baseline": None,
        }
        self.rate_state = {
            "window_start_time": time.time(),
            "blink_count_window": 0,
            "last_window_rate": 0.0,
            "rate_history": [],
        }
        self.alert_state = {"low_start_time": None, "last_alert_time": None}
        self.prev_face_center_x: Optional[float] = None
        self.prev_left_width: Optional[float] = None
        self.prev_right_width: Optional[float] = None
        self.avg_ear = 0.0
        self.eye_state_text = "OPEN"
        self.total_blinks = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        now = time.time()

        update_blink_rate(now, self.rate_state)
        live_blink_rate = compute_live_blink_rate_per_min(now, self.rate_state)
        last_window_rate = self.rate_state["last_window_rate"]

        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        face_list = getattr(results, "multi_face_landmarks", None)

        frame_ignored = False
        eyes_visible = False

        if face_list:
            face_landmarks = face_list[0]
            if has_required_landmarks(face_landmarks):
                draw_eye_landmarks(img, face_landmarks, LEFT_EYE_IDX)
                draw_eye_landmarks(img, face_landmarks, RIGHT_EYE_IDX)

                self.avg_ear = (
                    calculate_ear(face_landmarks, LEFT_EAR_IDX, w, h)
                    + calculate_ear(face_landmarks, RIGHT_EAR_IDX, w, h)
                ) / 2.0

                left_width = get_eye_width(face_landmarks, LEFT_EAR_IDX, w, h)
                right_width = get_eye_width(face_landmarks, RIGHT_EAR_IDX, w, h)
                face_center_x, face_width = get_face_center_and_width(face_landmarks, w)

                face_confident = face_width >= MIN_FACE_WIDTH_PX
                face_stable = True
                if self.prev_face_center_x is not None and face_width > 0:
                    face_stable = (
                        abs(face_center_x - self.prev_face_center_x)
                        <= FACE_MOVE_THRESHOLD_RATIO * face_width
                    )

                eyes_big_enough = left_width >= MIN_EYE_WIDTH_PX and right_width >= MIN_EYE_WIDTH_PX
                eyes_stable = True
                if self.prev_left_width and self.prev_right_width:
                    left_change = abs(left_width - self.prev_left_width) / self.prev_left_width
                    right_change = abs(right_width - self.prev_right_width) / self.prev_right_width
                    eyes_stable = (
                        left_change <= MAX_EYE_WIDTH_CHANGE_RATIO
                        and right_change <= MAX_EYE_WIDTH_CHANGE_RATIO
                    )

                if not face_confident or not face_stable:
                    frame_ignored = True
                else:
                    eyes_visible = eyes_big_enough and eyes_stable

                self.prev_face_center_x = face_center_x
                self.prev_left_width = left_width
                self.prev_right_width = right_width
            else:
                frame_ignored = True
        else:
            self.blink_state["eye_closed"] = False
            self.blink_state["closed_frame_count"] = 0
            self.prev_face_center_x = None
            self.prev_left_width = None
            self.prev_right_width = None

        blink_threshold = EAR_THRESHOLD
        if eyes_visible:
            if self.blink_state["ear_baseline"] is None:
                self.blink_state["ear_baseline"] = self.avg_ear
            else:
                self.blink_state["ear_baseline"] = (
                    (1.0 - EAR_BASELINE_ALPHA) * self.blink_state["ear_baseline"]
                    + EAR_BASELINE_ALPHA * self.avg_ear
                )
            blink_threshold = max(
                MIN_DYNAMIC_EAR_THRESHOLD,
                self.blink_state["ear_baseline"] * EAR_DYNAMIC_RATIO,
            )

        if not frame_ignored:
            blink_counted, self.eye_state_text = detect_blink(
                self.avg_ear,
                blink_threshold,
                eyes_visible,
                now,
                self.blink_state,
            )
            if blink_counted:
                self.total_blinks += 1
                self.rate_state["blink_count_window"] += 1
        else:
            self.blink_state["eye_closed"] = False
            self.blink_state["closed_frame_count"] = 0
            self.eye_state_text = "OPEN"

        status_text, _, cd_rem, low_dur = check_alert(now, live_blink_rate, self.alert_state)
        status_color = (0, 0, 255) if status_text == STATUS_ALERT else (0, 255, 0)
        frame_tag = "SKIPPED" if frame_ignored else "VALID"
        suffix = f" (cd {cd_rem:.0f}s)" if cd_rem > 0 else ""

        lines = [
            f"EAR: {self.avg_ear:.3f}",
            f"EAR threshold: {blink_threshold:.3f}",
            f"Blink count (window): {self.rate_state['blink_count_window']}",
            f"Total blinks: {self.total_blinks}",
            f"Blink rate live: {live_blink_rate:.1f}/min",
            f"Blink rate last window: {last_window_rate:.1f}/min",
            f"Low blink duration: {low_dur:.1f}s",
            f"Frame: {frame_tag}",
            f"Eye state: {self.eye_state_text}",
        ]
        y = 30
        for t in lines:
            cv2.putText(img, t, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 26
        cv2.putText(img, status_text + suffix, (15, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Blink Rate Eye Strain Monitor", layout="wide")
st.title("Blink Rate Eye Strain Monitor")
st.caption("Real-time blink detection with EAR + MediaPipe Face Mesh")
st.info("Allow webcam access. Blink naturally and keep face centered for best tracking.")

webrtc_streamer(
    key="blink-monitor",
    video_processor_factory=BlinkProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
