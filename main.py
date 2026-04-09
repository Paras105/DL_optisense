"""Robust blink monitor: MediaPipe Face Mesh + EAR + stable alerting.

Key behavior:
- Blink detection only on stable, clearly visible eyes.
- Blink counted on CLOSED->OPEN transition with cooldown.
- Fixed-size time window for blink-rate updates.
- Sustained low-rate alert with cooldown + beep.
"""
import math
import os
import sys
import time
import types
from pathlib import Path

import cv2

# MediaPipe top-level import can pull tasks->tensorflow chain in some envs.
# We only need solutions.face_mesh, so we stub tasks safely.
if "mediapipe.tasks.python" not in sys.modules:
    _mp_tasks = types.ModuleType("mediapipe.tasks")
    _mp_tasks_py = types.ModuleType("mediapipe.tasks.python")
    setattr(_mp_tasks, "python", _mp_tasks_py)
    sys.modules["mediapipe.tasks"] = _mp_tasks
    sys.modules["mediapipe.tasks.python"] = _mp_tasks_py

from mediapipe.python.solutions import face_mesh as mp_face_mesh

# -------------------------
# Tunable config (easy edit)
# -------------------------
EAR_THRESHOLD = 0.21
CLOSED_FRAMES_REQUIRED = 3
BLINK_COOLDOWN_SEC = 0.25
MIN_DYNAMIC_EAR_THRESHOLD = 0.18
EAR_BASELINE_ALPHA = 0.08
EAR_DYNAMIC_RATIO = 0.75

# Change this one variable if you want 15 minutes etc.
RATE_WINDOW_SEC = 60.0
LOW_BLINK_THRESHOLD_PER_MIN = 12.0
LOW_BLINK_PERSIST_SEC = 30.0
ALERT_COOLDOWN_SEC = 60.0

MIN_FACE_WIDTH_PX = 120
FACE_MOVE_THRESHOLD_RATIO = 0.20
MIN_EYE_WIDTH_PX = 12
MAX_EYE_WIDTH_CHANGE_RATIO = 0.55

CNN_H5_NAME = "eye_model.h5"
ENV_MODEL_PATH = "EYE_MODEL_PATH"
CNN_INPUT_SIZE = 64

STATUS_OK = "Blink rate normal"
STATUS_ALERT = "LOW BLINK RATE - TAKE A BREAK"

# MediaPipe indices
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
ALL_EYE_OUTLINE_IDX = list(dict.fromkeys(LEFT_EYE_IDX + RIGHT_EYE_IDX))
MAX_REQUIRED_IDX = max(max(LEFT_EYE_IDX), max(RIGHT_EYE_IDX), max(LEFT_EAR_IDX), max(RIGHT_EAR_IDX))


def beep_alert():
    """Cross-platform beep: winsound on Windows, terminal bell fallback elsewhere."""
    try:
        import winsound

        winsound.Beep(1200, 300)
    except Exception:
        print("\a", end="")
        print("[ALERT] beep")


def get_point(face_landmarks, idx, w, h):
    lm = face_landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)


def draw_eye_landmarks(frame, face_landmarks, eye_indices):
    h, w = frame.shape[:2]
    for idx in eye_indices:
        x, y = get_point(face_landmarks, idx, w, h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def calculate_ear(face_landmarks, ear_indices, w, h):
    p = [get_point(face_landmarks, ear_indices[i], w, h) for i in range(6)]
    horiz = math.hypot(p[0][0] - p[3][0], p[0][1] - p[3][1])
    if horiz == 0:
        return 0.0
    v1 = math.hypot(p[1][0] - p[5][0], p[1][1] - p[5][1])
    v2 = math.hypot(p[2][0] - p[4][0], p[2][1] - p[4][1])
    return (v1 + v2) / (2.0 * horiz)


def get_eye_width(face_landmarks, ear_indices, w, h):
    p1 = get_point(face_landmarks, ear_indices[0], w, h)
    p4 = get_point(face_landmarks, ear_indices[3], w, h)
    return math.hypot(p1[0] - p4[0], p1[1] - p4[1])


def get_face_center_and_width(face_landmarks, w):
    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
    if not xs:
        return 0.0, 0.0
    left, right = min(xs), max(xs)
    return (left + right) / 2.0, float(right - left)


def has_required_landmarks(face_landmarks):
    return len(face_landmarks.landmark) > MAX_REQUIRED_IDX


def detect_blink(avg_ear, blink_threshold, eyes_visible, now, state):
    """State machine: CLOSED after N frames below threshold; count only CLOSED->OPEN with cooldown."""
    if not eyes_visible:
        state["eye_closed"] = False
        state["closed_frame_count"] = 0
        return False, "OPEN"

    if avg_ear < blink_threshold:
        state["closed_frame_count"] += 1
        if state["closed_frame_count"] >= CLOSED_FRAMES_REQUIRED:
            state["eye_closed"] = True
        return False, "CLOSED" if state["eye_closed"] else "OPEN"

    # EAR back above threshold => possible blink transition
    blink_counted = False
    if state["eye_closed"]:
        if (now - state["last_blink_time"]) >= BLINK_COOLDOWN_SEC:
            blink_counted = True
            state["last_blink_time"] = now
    state["eye_closed"] = False
    state["closed_frame_count"] = 0
    return blink_counted, "OPEN"


def update_blink_rate(now, rate_state):
    """Fixed window rate update. Every RATE_WINDOW_SEC, store rate and reset window counter."""
    while now - rate_state["window_start_time"] >= RATE_WINDOW_SEC:
        elapsed = RATE_WINDOW_SEC
        window_rate = (rate_state["blink_count_window"] * 60.0) / elapsed
        rate_state["last_window_rate"] = window_rate
        rate_state["rate_history"].append(window_rate)
        rate_state["blink_count_window"] = 0
        rate_state["window_start_time"] += RATE_WINDOW_SEC


def compute_live_blink_rate_per_min(current_time, rate_state):
    """Blinks/min from current partial window (real time), for alert threshold logic."""
    elapsed = current_time - rate_state["window_start_time"]
    if elapsed <= 1e-6:
        return rate_state["last_window_rate"]
    return (rate_state["blink_count_window"] * 60.0) / elapsed


def check_alert(current_time, blink_rate_per_min, alert_state):
    """Alert timing uses ONLY real time + blink rate — never invalidated by skipped frames.

    - When blink_rate < threshold: start/continue low_blink timer (wall clock).
    - Reset timer ONLY when blink_rate >= threshold.
    - Alert when duration >= LOW_BLINK_PERSIST_SEC (and cooldown allows beep).
    """
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


def resolve_eye_model_path():
    env_path = os.environ.get(ENV_MODEL_PATH, "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p
        print(f"[CNN] {ENV_MODEL_PATH} set but file not found: {p}")

    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir / CNN_H5_NAME, Path.cwd() / CNN_H5_NAME]
    for p in candidates:
        if p.is_file():
            return p.resolve()

    print("[CNN] Model not found. EAR-only mode.")
    return None


def try_load_cnn(model_path):
    if model_path is None:
        return None
    try:
        import keras
    except ImportError:
        print("[CNN] keras/tensorflow not available. EAR-only mode.")
        return None

    try:
        model = keras.models.load_model(str(model_path), compile=False, safe_mode=False)
    except TypeError:
        try:
            model = keras.models.load_model(str(model_path), compile=False)
        except Exception as e:
            print(f"[CNN] Could not load model ({e}). EAR-only mode.")
            return None
    except Exception as e:
        print(f"[CNN] Could not load model ({e}). EAR-only mode.")
        return None

    print(f"[CNN] Loaded: {model_path}")
    return model


def crop_eye_region(frame_bgr, face_landmarks, w, h):
    xs, ys = [], []
    for idx in ALL_EYE_OUTLINE_IDX:
        x, y = get_point(face_landmarks, idx, w, h)
        xs.append(x)
        ys.append(y)
    if not xs:
        return None

    min_x, max_x = max(0, min(xs)), min(w, max(xs))
    min_y, max_y = max(0, min(ys)), min(h, max(ys))
    pad_x = int((max_x - min_x) * 0.25) + 8
    pad_y = int((max_y - min_y) * 0.25) + 8
    min_x, max_x = max(0, min_x - pad_x), min(w, max_x + pad_x)
    min_y, max_y = max(0, min_y - pad_y), min(h, max_y + pad_y)

    if max_x - min_x < 16 or max_y - min_y < 16:
        return None
    return frame_bgr[min_y:max_y, min_x:max_x]


def cnn_open_prob(cnn_model, eye_crop_bgr):
    if cnn_model is None or eye_crop_bgr is None or eye_crop_bgr.size == 0:
        return None
    rgb = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (CNN_INPUT_SIZE, CNN_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = resized.astype("float32") / 255.0
    pred = cnn_model.predict_on_batch(x[None, ...])
    return float(pred.reshape(-1)[0])


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cnn_model = try_load_cnn(resolve_eye_model_path())
    cnn_active = cnn_model is not None

    blink_state = {
        "eye_closed": False,
        "closed_frame_count": 0,
        "last_blink_time": 0.0,
        "ear_baseline": None,
    }
    rate_state = {
        "window_start_time": time.time(),
        "blink_count_window": 0,
        "last_window_rate": 0.0,
        "rate_history": [],
    }
    alert_state = {
        "low_start_time": None,
        "last_alert_time": None,
    }

    total_blinks = 0
    avg_ear = 0.0
    eye_state_text = "OPEN"
    status_text = STATUS_OK
    status_color = (0, 255, 0)
    prev_face_center_x = None
    prev_left_width = None
    prev_right_width = None
    cnn_debug_text = ""

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            current_time = time.time()
            h, w = frame.shape[:2]
            cnn_debug_text = ""

            update_blink_rate(current_time, rate_state)
            blink_rate_last_window = rate_state["last_window_rate"]
            live_blink_rate_per_min = compute_live_blink_rate_per_min(current_time, rate_state)

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_landmarks_list = getattr(results, "multi_face_landmarks", None)

            eyes_visible = False
            frame_ignored = False

            if face_landmarks_list:
                face_landmarks = face_landmarks_list[0]

                if has_required_landmarks(face_landmarks):
                    draw_eye_landmarks(frame, face_landmarks, LEFT_EYE_IDX)
                    draw_eye_landmarks(frame, face_landmarks, RIGHT_EYE_IDX)

                    avg_ear = (
                        calculate_ear(face_landmarks, LEFT_EAR_IDX, w, h)
                        + calculate_ear(face_landmarks, RIGHT_EAR_IDX, w, h)
                    ) / 2.0

                    left_width = get_eye_width(face_landmarks, LEFT_EAR_IDX, w, h)
                    right_width = get_eye_width(face_landmarks, RIGHT_EAR_IDX, w, h)

                    face_center_x, face_width = get_face_center_and_width(face_landmarks, w)
                    face_confident = face_width >= MIN_FACE_WIDTH_PX

                    face_stable = True
                    if prev_face_center_x is not None and face_width > 0:
                        face_stable = (
                            abs(face_center_x - prev_face_center_x)
                            <= FACE_MOVE_THRESHOLD_RATIO * face_width
                        )

                    eyes_big_enough = left_width >= MIN_EYE_WIDTH_PX and right_width >= MIN_EYE_WIDTH_PX
                    eyes_stable = True
                    if prev_left_width and prev_right_width:
                        left_change = abs(left_width - prev_left_width) / prev_left_width
                        right_change = abs(right_width - prev_right_width) / prev_right_width
                        eyes_stable = (
                            left_change <= MAX_EYE_WIDTH_CHANGE_RATIO
                            and right_change <= MAX_EYE_WIDTH_CHANGE_RATIO
                        )

                    if not face_confident or not face_stable:
                        frame_ignored = True
                    else:
                        eyes_visible = eyes_big_enough and eyes_stable

                    prev_face_center_x = face_center_x
                    prev_left_width = left_width
                    prev_right_width = right_width

                    if cnn_active and eyes_visible:
                        crop = crop_eye_region(frame, face_landmarks, w, h)
                        p_open = cnn_open_prob(cnn_model, crop)
                        if p_open is not None:
                            cnn_debug_text = f"CNN open prob: {p_open:.2f}"
                else:
                    # Missing landmarks -> ignore this frame for blink logic
                    frame_ignored = True
            else:
                # Face disappeared: reset only temporary detection states
                blink_state["eye_closed"] = False
                blink_state["closed_frame_count"] = 0
                prev_face_center_x = None
                prev_left_width = None
                prev_right_width = None

            blink_threshold = EAR_THRESHOLD
            if eyes_visible:
                if blink_state["ear_baseline"] is None:
                    blink_state["ear_baseline"] = avg_ear
                else:
                    blink_state["ear_baseline"] = (
                        (1.0 - EAR_BASELINE_ALPHA) * blink_state["ear_baseline"]
                        + EAR_BASELINE_ALPHA * avg_ear
                    )
                blink_threshold = max(
                    MIN_DYNAMIC_EAR_THRESHOLD,
                    blink_state["ear_baseline"] * EAR_DYNAMIC_RATIO,
                )

            # Blink detection only on valid (non-skipped) frames — alert uses real time, always
            if not frame_ignored:
                blink_counted, eye_state_text = detect_blink(
                    avg_ear,
                    blink_threshold,
                    eyes_visible,
                    current_time,
                    blink_state,
                )
                if blink_counted:
                    total_blinks += 1
                    rate_state["blink_count_window"] += 1
            else:
                blink_state["eye_closed"] = False
                blink_state["closed_frame_count"] = 0
                eye_state_text = "OPEN"

            status_text, should_beep, cd_rem, low_duration_sec = check_alert(
                current_time,
                live_blink_rate_per_min,
                alert_state,
            )
            if should_beep:
                beep_alert()
                print(
                    f"[ALERT] low blink sustained. rate={live_blink_rate_per_min:.1f}/min, "
                    f"cooldown={ALERT_COOLDOWN_SEC:.0f}s"
                )

            status_color = (0, 0, 255) if status_text == STATUS_ALERT else (0, 255, 0)
            alert_suffix = f" (cd {cd_rem:.0f}s)" if cd_rem > 0 else ""

            frame_tag = "SKIPPED (blink detection off)" if frame_ignored else "VALID"
            # Debug overlay
            lines = [
                f"EAR: {avg_ear:.3f}",
                f"EAR threshold: {blink_threshold:.3f}",
                f"Blink count (window): {rate_state['blink_count_window']}",
                f"Total blinks: {total_blinks}",
                f"Blink rate est. (live): {live_blink_rate_per_min:.1f}/min",
                f"Blink rate (last closed window): {blink_rate_last_window:.1f}/min",
                f"Low blink duration: {low_duration_sec:.1f}s",
                f"Frame: {frame_tag}",
                f"Eye state: {eye_state_text}",
            ]

            y = 35
            for text in lines:
                cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y += 32

            cv2.putText(
                frame,
                status_text + alert_suffix,
                (20, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2,
            )

            hud = "CNN: ACTIVE" if cnn_active else "CNN: OFF (EAR only)"
            cv2.putText(frame, hud, (20, y + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 220, 160), 1)
            if cnn_debug_text:
                cv2.putText(frame, cnn_debug_text, (20, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            cv2.imshow("Blink Rate Eye Strain Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
