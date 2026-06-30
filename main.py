"""
=============================================================================
main.py — Real-Time People Counter (Entry Point)
=============================================================================
OpenCV modules demonstrated end-to-end:

  Module        | Where used
  ------------- | ----------------------------------------------------------
  core          | Matrix slicing, addWeighted, countNonZero, imencode
  videoio       | VideoCapture – open webcam, read frames, release
  highgui       | namedWindow, imshow, waitKey, destroyAllWindows
  imgproc       | cvtColor, GaussianBlur, resize, rectangle, putText
  imgcodecs     | imwrite (save_frame in utils.py)
  features2d    | ORB detector & drawKeypoints (utils.py)
  objdetect     | CascadeClassifier Haar face detector (utils.py)
  dnn           | MobileNet-SSD blob creation, forward pass, NMS (detector.py)

Key controls
  Q  – quit
  S  – save current frame to saved_frames/
  O  – toggle ORB keypoint overlay
  F  – toggle face detection overlay
=============================================================================
"""

import sys
import time
import cv2
import numpy as np

# ── Local modules ─────────────────────────────────────────────────────────────
from detector import PersonDetector
from utils import (
    create_orb_detector,
    detect_orb_keypoints,
    draw_orb_overlay,
    load_face_cascade,
    detect_faces,
    draw_faces,
    draw_person_box,
    draw_hud,
    save_frame,
)


# =============================================================================
# Configuration constants
# =============================================================================
WINDOW_TITLE      = "Real-Time People Counter | Press Q to quit"
CAMERA_INDEX      = 0          # 0 = default system webcam
FRAME_WIDTH       = 1280       # desired capture width  (px)
FRAME_HEIGHT      = 720        # desired capture height (px)
CONF_THRESHOLD    = 0.35       # minimum detection confidence (VOC MobileNet-SSD)
NMS_THRESHOLD     = 0.40       # NMS IoU overlap threshold
DETECT_EVERY_N    = 2          # run DNN every N frames for performance


# =============================================================================
# Webcam initialisation (videoio module)
# =============================================================================

def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """
    Open the webcam using cv2.VideoCapture (videoio).
    Tries multiple backends so it works on most Windows setups.
    Raises RuntimeError if no camera is available or only black frames come in.
    """
    # videoio – try backends in priority order:
    #   CAP_MSMF  = Media Foundation (best on Windows 10/11)
    #   CAP_DSHOW = DirectShow (legacy fallback)
    #   default   = let OpenCV auto-select
    backends = [
        (cv2.CAP_MSMF,  "MSMF (Media Foundation)"),
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_ANY,   "Auto-detect"),
    ]

    cap = None
    for backend_id, backend_name in backends:
        candidate = cv2.VideoCapture(index, backend_id)
        if not candidate.isOpened():
            candidate.release()
            continue

        # videoio – configure resolution & FPS
        candidate.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        candidate.set(cv2.CAP_PROP_FPS, 30)
        # NOTE: do NOT set BUFFERSIZE=1 – it causes black frames on many
        # Windows cameras by starving the internal frame queue.

        # Warm-up: Windows cameras return black frames for the first
        # several reads while the sensor exposure auto-adjusts.
        print(f"[Camera] Trying backend: {backend_name} …")
        real_frame = False
        for _ in range(60):                        # up to 60 warm-up reads
            ret, test = candidate.read()
            if ret and test is not None:
                # Accept the frame only if it is not mostly black
                if test.mean() > 5.0:             # mean pixel > 5/255
                    real_frame = True
                    break

        if real_frame:
            actual_w = int(candidate.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(candidate.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Camera] Opened index {index} via {backend_name}  |  {actual_w}x{actual_h}")
            cap = candidate
            break
        else:
            print(f"[Camera] {backend_name} returned only black frames — trying next backend.")
            candidate.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera (index {index}) with any backend.\n"
            "Make sure a webcam is connected and not used by another app."
        )
    return cap


# =============================================================================
# Main loop
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("  Real-Time People Counter — OpenCV")
    print("=" * 60)

    # ── Step 1 : open webcam (videoio) ────────────────────────────────────
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as err:
        print(f"[ERROR] {err}")
        sys.exit(1)

    # ── Step 2 : load DNN person detector (dnn) ───────────────────────────
    print("[DNN] Loading MobileNet-SSD …")
    try:
        detector = PersonDetector(
            conf_threshold=CONF_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            use_gpu=False          # set True if CUDA is available
        )
        print("[DNN] Model loaded successfully.")
    except FileNotFoundError as err:
        print(f"[ERROR] {err}")
        cap.release()
        sys.exit(1)

    # ── Step 3 : initialise ORB feature detector (features2d) ────────────
    orb = create_orb_detector(n_features=300)
    orb_enabled  = False          # toggled with 'O' key
    orb_kp_count = 0

    # ── Step 4 : load Haar cascade for face detection (objdetect) ─────────
    print("[Haar] Loading face cascade …")
    try:
        face_cascade = load_face_cascade()
        face_enabled = False      # toggled with 'F' key
        face_count   = 0
        print("[Haar] Cascade loaded.")
    except RuntimeError as err:
        print(f"[WARNING] {err}  — face detection disabled.")
        face_cascade = None
        face_enabled = False
        face_count   = 0

    # ── Step 5 : create HighGUI window (highgui) ──────────────────────────
    # highgui – creates a named, resizable output window
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, FRAME_WIDTH, FRAME_HEIGHT)

    # ── Session state ─────────────────────────────────────────────────────
    max_people   = 0
    frame_count  = 0
    people_dets  = []            # cached detections from last DNN run
    fps          = 0.0
    fps_timer    = time.perf_counter()
    fps_frames   = 0

    print("[INFO] Streaming — Q: quit | S: save | O: toggle ORB | F: toggle faces")
    print("=" * 60)

    # ======================================================================
    # Main capture / display loop
    # ======================================================================
    while True:
        # videoio – read next frame from the webcam
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Failed to grab frame — retrying ...")
            time.sleep(0.05)
            continue

        # Guard: skip genuinely black frames (sensor not ready)
        if frame.mean() < 3.0:
            continue

        frame_count += 1
        fps_frames  += 1

        # ── FPS calculation ───────────────────────────────────────────────
        now = time.perf_counter()
        elapsed = now - fps_timer
        if elapsed >= 0.5:                     # update display twice/sec
            fps       = fps_frames / elapsed
            fps_timer  = now
            fps_frames = 0

        # ── imgproc – optional slight denoise for better detection ────────
        # Gaussian blur is run on a downscaled copy (not the display frame)
        # to reduce noise before feeding to the DNN.
        small = cv2.resize(frame, (640, 360))
        small = cv2.GaussianBlur(small, (3, 3), 0)

        # ── DNN inference (every DETECT_EVERY_N frames for performance) ───
        if frame_count % DETECT_EVERY_N == 0:
            people_dets = detector.detect(frame)

        # ── Session maximum tracker ───────────────────────────────────────
        people_count = len(people_dets)
        if people_count > max_people:
            max_people = people_count

        # ── Draw person bounding boxes (imgproc + utils) ──────────────────
        for i, det in enumerate(people_dets, start=1):
            x1, y1, x2, y2 = det["box"]
            draw_person_box(frame, x1, y1, x2, y2, det["confidence"], i)

        # ── ORB keypoint overlay (features2d) — toggled by 'O' ───────────
        if orb_enabled:
            kps, _ = detect_orb_keypoints(frame, orb)
            orb_kp_count = len(kps)
            frame = draw_orb_overlay(frame, kps, alpha=0.35)
        else:
            orb_kp_count = 0

        # ── Face detection overlay (objdetect) — toggled by 'F' ──────────
        if face_enabled and face_cascade is not None:
            # Run on downscaled frame for speed, then scale boxes back up
            scale_x = frame.shape[1] / small.shape[1]
            scale_y = frame.shape[0] / small.shape[0]
            raw_faces = detect_faces(small, face_cascade)
            scaled_faces = [
                (int(x * scale_x), int(y * scale_y),
                 int(w * scale_x), int(h * scale_y))
                for (x, y, w, h) in raw_faces
            ]
            face_count = len(scaled_faces)
            frame = draw_faces(frame, scaled_faces)
        else:
            face_count = 0

        # ── HUD overlay (imgproc, core) ───────────────────────────────────
        frame = draw_hud(
            frame,
            people_count=people_count,
            max_people=max_people,
            fps=fps,
            orb_kp_count=orb_kp_count,
            face_count=face_count,
            orb_enabled=orb_enabled,
            face_enabled=face_enabled,
        )

        # ── highgui – display the annotated frame ─────────────────────────
        cv2.imshow(WINDOW_TITLE, frame)

        # ── highgui – keyboard event handling (1 ms poll) ─────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            # Q – graceful shutdown
            print(f"\n[INFO] Quit — session max people: {max_people}")
            break

        elif key == ord("s") or key == ord("S"):
            # S – imgcodecs save frame
            try:
                saved_path = save_frame(frame, save_dir="saved_frames")
                print(f"[SAVE] Frame saved → {saved_path}")
            except IOError as err:
                print(f"[ERROR] Could not save frame: {err}")

        elif key == ord("o") or key == ord("O"):
            # O – toggle ORB keypoint overlay
            orb_enabled = not orb_enabled
            state = "ON" if orb_enabled else "OFF"
            print(f"[ORB] Keypoint overlay {state}")

        elif key == ord("f") or key == ord("F"):
            # F – toggle Haar face detection
            if face_cascade is not None:
                face_enabled = not face_enabled
                state = "ON" if face_enabled else "OFF"
                print(f"[FACE] Face detection {state}")
            else:
                print("[FACE] Face detection unavailable — cascade not loaded.")

        # Also quit if the window 'X' button is pressed
        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Window closed by user.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────
    # videoio – release the camera device
    cap.release()
    # highgui – destroy all HighGUI windows
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Goodbye!")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    main()
