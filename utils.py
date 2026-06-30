"""
=============================================================================
utils.py — Utility helpers for the Real-Time People Counter
=============================================================================
Modules demonstrated here:
  • core      – matrix / array operations via NumPy wrappers
  • imgproc   – drawing, colour conversion, Gaussian blur, resize
  • features2d – ORB keypoint detection & descriptor computation
  • objdetect – Haar Cascade face detection
=============================================================================
"""

import cv2
import numpy as np
import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Module: features2d  –  ORB (Oriented FAST and Rotated BRIEF) keypoints
# ---------------------------------------------------------------------------
def create_orb_detector(n_features: int = 300) -> cv2.ORB:
    """
    Create and return an ORB feature detector.
    ORB is a fast, rotation-invariant binary descriptor shipped inside
    the features2d module.
    """
    return cv2.ORB_create(nfeatures=n_features)


def detect_orb_keypoints(frame: np.ndarray,
                          orb: cv2.ORB) -> tuple[np.ndarray, list]:
    """
    Detect ORB keypoints on a greyscale version of *frame*.

    Returns
    -------
    keypoints : list[cv2.KeyPoint]
    descriptors : np.ndarray or None
    """
    # imgproc – colour-space conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def draw_orb_overlay(frame: np.ndarray,
                     keypoints: list,
                     alpha: float = 0.35) -> np.ndarray:
    """
    Blend a semi-transparent keypoint layer onto *frame*.
    Uses cv2.drawKeypoints (features2d) and cv2.addWeighted (core/imgproc).
    """
    kp_layer = np.zeros_like(frame)
    cv2.drawKeypoints(
        frame, keypoints, kp_layer,
        color=(0, 215, 255),                       # golden-amber dots
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    # core – weighted blend of two matrices
    return cv2.addWeighted(frame, 1.0, kp_layer, alpha, 0)


# ---------------------------------------------------------------------------
# Module: objdetect  –  Haar Cascade face detector
# ---------------------------------------------------------------------------
def load_face_cascade(xml_path: str | None = None) -> cv2.CascadeClassifier:
    """
    Load the frontal-face Haar Cascade bundled with OpenCV.
    objdetect module is used internally by CascadeClassifier.
    """
    if xml_path is None:
        # Use the cascade file that ships with the opencv package
        xml_path = str(
            Path(cv2.__file__).parent /
            "data" / "haarcascade_frontalface_default.xml"
        )
    cascade = cv2.CascadeClassifier(xml_path)
    if cascade.empty():
        raise RuntimeError(
            f"Failed to load Haar cascade from: {xml_path}\n"
            "Make sure opencv-contrib-python is installed."
        )
    return cascade


def detect_faces(frame: np.ndarray,
                 cascade: cv2.CascadeClassifier) -> list:
    """
    Run face detection on *frame* using the supplied Haar cascade.
    Returns a list of (x, y, w, h) rectangles.
    """
    # imgproc – convert to greyscale for the cascade detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # imgproc – equalise histogram to improve detection in varied lighting
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces if len(faces) > 0 else []


def draw_faces(frame: np.ndarray, faces: list) -> np.ndarray:
    """
    Draw cyan bounding boxes around detected faces.
    Uses imgproc drawing primitives.
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (255, 215, 0), 1)           # cyan-gold face box
        cv2.putText(frame, "face", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 215, 0), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Module: dnn  –  MobileNet-SSD person detection (see detector.py)
# ---------------------------------------------------------------------------
# (Loading / inference live in detector.py; drawing helpers are here.)

def draw_person_box(frame: np.ndarray,
                    x1: int, y1: int, x2: int, y2: int,
                    confidence: float,
                    idx: int) -> np.ndarray:
    """
    Draw a green bounding box + confidence label for one detected person.

    Parameters
    ----------
    frame      : BGR image array (core)
    x1,y1,x2,y2 : box corners (already clamped to frame bounds)
    confidence : detection confidence 0–1
    idx        : person index shown on the label
    """
    # imgproc – rectangle primitive
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)

    label = f"#{idx}  {confidence:.0%}"
    lbl_w, lbl_h = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]

    # Filled background for readability (core – ROI slice assignment)
    cv2.rectangle(frame,
                  (x1, max(y1 - lbl_h - 8, 0)),
                  (x1 + lbl_w + 4, y1),
                  (0, 180, 0), cv2.FILLED)
    cv2.putText(frame, label,
                (x1 + 2, max(y1 - 4, lbl_h)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Module: imgproc  –  HUD overlay helpers
# ---------------------------------------------------------------------------
def crowd_status(count: int) -> tuple[str, tuple]:
    """
    Return (status_string, colour_BGR) based on the current people count.
    """
    if count <= 3:
        return "LOW CROWD", (0, 200, 80)
    elif count <= 8:
        return "MEDIUM CROWD", (0, 180, 255)
    else:
        return "HIGH CROWD", (0, 0, 230)


def draw_hud(frame: np.ndarray,
             people_count: int,
             max_people: int,
             fps: float,
             orb_kp_count: int,
             face_count: int,
             orb_enabled: bool,
             face_enabled: bool) -> np.ndarray:
    """
    Render the complete Heads-Up Display onto *frame*.

    Panels
    ------
    Top-left  : people count, crowd status, max people
    Top-right : FPS, resolution, date/time
    Bottom-left : module indicators (ORB kp count, face count)
    """
    h, w = frame.shape[:2]
    now = datetime.datetime.now()

    # ── semi-transparent dark panel (top-left) ──────────────────────────
    panel_h, panel_w = 145, 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h),
                  (15, 15, 15), cv2.FILLED)
    # core – addWeighted blends the panel with the underlying frame
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    status_str, status_color = crowd_status(people_count)

    _put(frame, f"PEOPLE COUNT : {people_count}",
         (18, 34), scale=0.70, color=(255, 255, 255), thickness=2)
    _put(frame, status_str,
         (18, 60), scale=0.65, color=status_color, thickness=2)
    _put(frame, f"SESSION MAX  : {max_people}",
         (18, 84), scale=0.55, color=(180, 180, 180))
    _put(frame, f"ORB KP : {orb_kp_count if orb_enabled else 'OFF'}",
         (18, 106), scale=0.50, color=(0, 215, 255))
    _put(frame, f"FACES  : {face_count if face_enabled else 'OFF'}",
         (18, 126), scale=0.50, color=(255, 215, 0))
    _put(frame, f"Q:Quit  S:Save  O:ORB  F:Face",
         (18, 148), scale=0.42, color=(130, 130, 130))

    # ── semi-transparent dark panel (top-right) ─────────────────────────
    r_panel_w, r_panel_h = 235, 90
    overlay2 = frame.copy()
    cv2.rectangle(overlay2,
                  (w - r_panel_w - 8, 8),
                  (w - 8, 8 + r_panel_h),
                  (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)

    _put(frame, f"FPS : {fps:5.1f}",
         (w - r_panel_w, 30), scale=0.60, color=(100, 255, 100), thickness=2)
    _put(frame, f"{w} x {h}",
         (w - r_panel_w, 54), scale=0.55, color=(180, 180, 180))
    _put(frame, now.strftime("%Y-%m-%d  %H:%M:%S"),
         (w - r_panel_w, 78), scale=0.48, color=(140, 200, 255))

    return frame


def _put(frame: np.ndarray,
         text: str,
         origin: tuple,
         scale: float = 0.55,
         color: tuple = (255, 255, 255),
         thickness: int = 1) -> None:
    """Thin wrapper around cv2.putText using anti-aliased rendering."""
    cv2.putText(frame, text, origin,
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Module: imgcodecs  –  frame saving
# ---------------------------------------------------------------------------
def save_frame(frame: np.ndarray, save_dir: str = "saved_frames") -> str:
    """
    Save *frame* as a JPEG file inside *save_dir*.
    Uses cv2.imwrite (imgcodecs module).

    Returns the saved file path.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = str(Path(save_dir) / f"capture_{ts}.jpg")
    # imgcodecs – encode and write image to disk
    params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    success = cv2.imwrite(path, frame, params)
    if not success:
        raise IOError(f"cv2.imwrite failed for path: {path}")
    return path
