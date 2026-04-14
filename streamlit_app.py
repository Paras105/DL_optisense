"""Streamlit UI for Blink Rate Eye Strain Monitor.

Run:
    streamlit run streamlit_app.py
"""
import math
import importlib
import time
from typing import Optional

import av  # pyright: ignore[reportMissingImports]
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer  # pyright: ignore[reportMissingImports]

MEDIAPIPE_AVAILABLE = True
MEDIAPIPE_IMPORT_ERROR = ""

try:
    import mediapipe as mp
    _solutions = getattr(mp, "solutions", None)
    if _solutions is None:
        _solutions = importlib.import_module("mediapipe.solutions")
    mp_face_mesh = getattr(_solutions, "face_mesh", None)
    if mp_face_mesh is None:
        raise ImportError("mediapipe.solutions.face_mesh is unavailable")
except Exception as exc:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    mp_face_mesh = None

EAR_THRESHOLD = 0.21
CLOSED_FRAMES_REQUIRED = 2
BLINK_COOLDOWN_SEC = 0.25
MIN_DYNAMIC_EAR_THRESHOLD = 0.19
EAR_BASELINE_ALPHA = 0.08
EAR_DYNAMIC_RATIO = 0.85
EAR_CLOSE_RATIO = 0.98
EAR_OPEN_RATIO = 1.05
MAX_SKIP_FRAMES_FOR_BLINK_HOLD = 2

RATE_WINDOW_SEC = 60.0
LOW_BLINK_THRESHOLD_PER_MIN = 12.0
LOW_BLINK_PERSIST_SEC = 30.0
ALERT_COOLDOWN_SEC = 60.0

MIN_FACE_WIDTH_PX = 80
FACE_MOVE_THRESHOLD_RATIO = 0.35
MIN_EYE_WIDTH_PX = 12
MAX_EYE_WIDTH_CHANGE_RATIO = 0.55

STATUS_OK = "Blink rate normal"
STATUS_ALERT = "LOW BLINK RATE - TAKE A BREAK"

LEFT_EYE_IDX = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_IDX = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]
LEFT_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EAR_IDX = [362, 385, 387, 263, 373, 380]
MAX_REQUIRED_IDX = max(max(LEFT_EYE_IDX), max(RIGHT_EYE_IDX), max(LEFT_EAR_IDX), max(RIGHT_EAR_IDX))


def get_point(face_landmarks, idx: int, w: int, h: int) -> tuple[int, int]:
    lm = face_landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)


def draw_eye_landmarks(draw: ImageDraw.ImageDraw, face_landmarks, eye_indices, w: int, h: int):
    for idx in eye_indices:
        x, y = get_point(face_landmarks, idx, w, h)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 255, 0))


def calculate_ear(face_landmarks, ear_indices, w: int, h: int) -> float:
    p = [get_point(face_landmarks, ear_indices[i], w, h) for i in range(6)]
    horiz = math.hypot(p[0][0] - p[3][0], p[0][1] - p[3][1])
    if horiz == 0:
        return 0.0
    v1 = math.hypot(p[1][0] - p[5][0], p[1][1] - p[5][1])
    v2 = math.hypot(p[2][0] - p[4][0], p[2][1] - p[4][1])
    return (v1 + v2) / (2.0 * horiz)


def get_eye_width(face_landmarks, ear_indices, w: int, h: int) -> float:
    p1 = get_point(face_landmarks, ear_indices[0], w, h)
    p4 = get_point(face_landmarks, ear_indices[3], w, h)
    return math.hypot(p1[0] - p4[0], p1[1] - p4[1])


def get_face_center_and_width(face_landmarks, w: int) -> tuple[float, float]:
    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
    if not xs:
        return 0.0, 0.0
    left, right = min(xs), max(xs)
    return (left + right) / 2.0, float(right - left)


def has_required_landmarks(face_landmarks) -> bool:
    return len(face_landmarks.landmark) > MAX_REQUIRED_IDX


def detect_blink(avg_ear: float, blink_threshold: float, eyes_visible: bool, now: float, state):
    if not eyes_visible:
        state["eye_closed"] = False
        state["closed_frame_count"] = 0
        return False, "OPEN"

    close_threshold = blink_threshold * EAR_CLOSE_RATIO
    open_threshold = blink_threshold * EAR_OPEN_RATIO

    if avg_ear < close_threshold:
        state["closed_frame_count"] += 1
        if state["closed_frame_count"] >= CLOSED_FRAMES_REQUIRED:
            state["eye_closed"] = True
        return False, "CLOSED" if state["eye_closed"] else "OPEN"

    blink_counted = False
    if state["eye_closed"] and avg_ear > open_threshold:
        if (now - state["last_blink_time"]) >= BLINK_COOLDOWN_SEC:
            blink_counted = True
            state["last_blink_time"] = now
        state["eye_closed"] = False
        state["closed_frame_count"] = 0
    elif not state["eye_closed"]:
        state["closed_frame_count"] = 0
    return blink_counted, "OPEN"


def update_blink_rate(now: float, rate_state):
    while now - rate_state["window_start_time"] >= RATE_WINDOW_SEC:
        rate_state["last_window_rate"] = (rate_state["blink_count_window"] * 60.0) / RATE_WINDOW_SEC
        rate_state["rate_history"].append(rate_state["last_window_rate"])
        rate_state["blink_count_window"] = 0
        rate_state["window_start_time"] += RATE_WINDOW_SEC


def compute_live_blink_rate_per_min(now: float, rate_state) -> float:
    elapsed = now - rate_state["window_start_time"]
    if elapsed <= 1e-6:
        return rate_state["last_window_rate"]
    return (rate_state["blink_count_window"] * 60.0) / elapsed


def check_alert(current_time: float, blink_rate_per_min: float, alert_state):
    if blink_rate_per_min < LOW_BLINK_THRESHOLD_PER_MIN:
        if alert_state["low_start_time"] is None:
            alert_state["low_start_time"] = current_time
    else:
        alert_state["low_start_time"] = None

    low_duration_sec = 0.0
    if alert_state["low_start_time"] is not None:
        low_duration_sec = current_time - alert_state["low_start_time"]

    sustained_low = low_duration_sec >= LOW_BLINK_PERSIST_SEC
    cooldown_remaining = 0.0
    if alert_state["last_alert_time"] is not None:
        cooldown_remaining = max(0.0, ALERT_COOLDOWN_SEC - (current_time - alert_state["last_alert_time"]))

    should_beep = False
    if sustained_low and cooldown_remaining <= 0.0:
        should_beep = True
        alert_state["last_alert_time"] = current_time
        cooldown_remaining = ALERT_COOLDOWN_SEC

    if sustained_low:
        return STATUS_ALERT, should_beep, cooldown_remaining, low_duration_sec
    return STATUS_OK, False, cooldown_remaining, low_duration_sec


class BlinkProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE and mp_face_mesh is not None:
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
            "skip_frame_streak": 0,
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
        img = np.ascontiguousarray(img[:, ::-1, :])
        h, w = img.shape[:2]
        now = time.time()
        pil = Image.fromarray(img[:, :, ::-1])
        draw = ImageDraw.Draw(pil)

        # Keep the app running even if mediapipe import/runtime init fails in cloud.
        if self.face_mesh is None:
            draw.text((15, 25), "Mediapipe unavailable in this runtime.", fill=(255, 80, 80))
            draw.text((15, 55), "Check deployment logs and requirements.", fill=(255, 180, 120))
            if MEDIAPIPE_IMPORT_ERROR:
                draw.text((15, 85), MEDIAPIPE_IMPORT_ERROR[:90], fill=(255, 180, 120))
            pil = pil.resize((960, 540))
            img = np.array(pil)[:, :, ::-1]
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        update_blink_rate(now, self.rate_state)
        live_blink_rate = compute_live_blink_rate_per_min(now, self.rate_state)
        last_window_rate = self.rate_state["last_window_rate"]

        results = self.face_mesh.process(img[:, :, ::-1])
        face_list = getattr(results, "multi_face_landmarks", None)

        frame_ignored = False
        eyes_visible = False

        if face_list:
            face_landmarks = face_list[0]
            if has_required_landmarks(face_landmarks):
                draw_eye_landmarks(draw, face_landmarks, LEFT_EYE_IDX, w, h)
                draw_eye_landmarks(draw, face_landmarks, RIGHT_EYE_IDX, w, h)

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
                    if not eyes_visible:
                        frame_ignored = True

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
                EAR_THRESHOLD,
                MIN_DYNAMIC_EAR_THRESHOLD,
                self.blink_state["ear_baseline"] * EAR_DYNAMIC_RATIO,
            )

        if not frame_ignored:
            self.blink_state["skip_frame_streak"] = 0
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
            self.blink_state["skip_frame_streak"] += 1
            if self.blink_state["skip_frame_streak"] > MAX_SKIP_FRAMES_FOR_BLINK_HOLD:
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
            draw.rectangle((8, y - 5, 365, y + 17), fill=(0, 0, 0))
            draw.text((15, y), t, fill=(255, 255, 0))
            y += 26
        draw.rectangle((8, y + 1, 430, y + 23), fill=(0, 0, 0))
        draw.text((15, y + 6), status_text + suffix, fill=(255, 80, 80) if status_text == STATUS_ALERT else (80, 255, 80))

        pil = pil.resize((960, 540))
        img = np.array(pil)[:, :, ::-1]
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Blink Rate Eye Strain Monitor", layout="centered")
st.title("Blink Rate Eye Strain Monitor")
st.caption("Real-time blink detection with EAR + MediaPipe Face Mesh")
st.info("Allow webcam access. Blink naturally and keep face centered for best tracking.")
if not MEDIAPIPE_AVAILABLE:
    st.error("Mediapipe failed to import in this environment.")
    st.code(MEDIAPIPE_IMPORT_ERROR or "Unknown import error")
st.markdown(
    """
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    [data-testid="stAppViewContainer"], .main {
        overflow-y: auto !important;
    }
    [data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

webrtc_streamer(
    key="blink-monitor",
    video_processor_factory=BlinkProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
