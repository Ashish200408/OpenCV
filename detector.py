"""
=============================================================================
detector.py — DNN-based person detection with MobileNet-SSD
=============================================================================
Module demonstrated: dnn
  • Load a Caffe / TF / ONNX model with cv2.dnn.readNet*
  • Build an input blob (blobFromImage)
  • Run a forward pass and parse detections
  • Non-maximum suppression via cv2.dnn.NMSBoxes
=============================================================================
"""

import cv2
import numpy as np
from pathlib import Path


# VOC class labels — the chuanqi305 MobileNet-SSD model is trained on
# PASCAL VOC (21 classes), NOT COCO (80 classes).
# IMPORTANT: person = index 15 in VOC (not 1 as in COCO)
VOC_LABELS = [
    "background",   # 0
    "aeroplane",    # 1
    "bicycle",      # 2
    "bird",         # 3
    "boat",         # 4
    "bottle",       # 5
    "bus",          # 6
    "car",          # 7
    "cat",          # 8
    "chair",        # 9
    "cow",          # 10
    "diningtable",  # 11
    "dog",          # 12
    "horse",        # 13
    "motorbike",    # 14
    "person",       # 15  <-- this is what we filter on
    "pottedplant",  # 16
    "sheep",        # 17
    "sofa",         # 18
    "train",        # 19
    "tvmonitor",    # 20
]

# Class index for "person" in the VOC label set
PERSON_CLASS_ID = 15

# ── Default model paths (relative to project root) ──────────────────────────
DEFAULT_PROTO  = "models/MobileNetSSD_deploy.prototxt"
DEFAULT_WEIGHTS = "models/MobileNetSSD_deploy.caffemodel"

# Input blob parameters for MobileNet-SSD
BLOB_SCALE   = 0.007843          # 1/127.5 — normalises pixels to [-1, 1]
BLOB_SIZE    = (300, 300)        # model's fixed input resolution
BLOB_MEAN    = 127.5             # mean subtraction


class PersonDetector:
    """
    Wraps a MobileNet-SSD network (dnn module) that detects persons.

    Parameters
    ----------
    proto   : path to the .prototxt model architecture file
    weights : path to the .caffemodel weight file
    conf_threshold : minimum detection confidence to keep
    nms_threshold  : IoU threshold for NMS (suppresses duplicates)
    use_gpu : attempt CUDA backend if True
    """

    def __init__(self,
                 proto: str = DEFAULT_PROTO,
                 weights: str = DEFAULT_WEIGHTS,
                 conf_threshold: float = 0.35,   # lower = catches more detections
                 nms_threshold: float = 0.40,
                 use_gpu: bool = False):

        self.conf_threshold = conf_threshold
        self.nms_threshold  = nms_threshold
        self._load_network(proto, weights, use_gpu)

    # ── Private helpers ──────────────────────────────────────────────────

    def _load_network(self, proto: str, weights: str, use_gpu: bool) -> None:
        """Load the Caffe model using cv2.dnn (dnn module)."""
        if not Path(proto).exists():
            raise FileNotFoundError(
                f"Model file not found: {proto}\n"
                "Run  python download_model.py  to fetch the weights."
            )
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"Weight file not found: {weights}\n"
                "Run  python download_model.py  to fetch the weights."
            )

        # dnn – load a Caffe model (architecture + pre-trained weights)
        self.net = cv2.dnn.readNetFromCaffe(proto, weights)

        if use_gpu:
            # dnn – switch to CUDA backend when a compatible GPU is present
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[DNN] Using CUDA backend.")
        else:
            # dnn – default: OpenCV's own optimised CPU backend
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[DNN] Using CPU backend.")

    # ── Public API ───────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run person detection on *frame*.

        Returns a list of dicts, one per detected person:
            { 'box': (x1, y1, x2, y2), 'confidence': float }

        Steps
        -----
        1. Build an input blob (dnn.blobFromImage)
        2. Feed blob to the network and call forward()
        3. Parse the SSD output tensor
        4. Apply NMS (dnn.NMSBoxes) to remove overlapping boxes
        5. Return only 'person' detections above conf_threshold
        """
        h, w = frame.shape[:2]

        # dnn – create a 4-D NCHW blob from the BGR image
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=BLOB_SCALE,
            size=BLOB_SIZE,
            mean=BLOB_MEAN,
            swapRB=False,          # Caffe uses BGR — no channel swap needed
            crop=False
        )

        # dnn – set the blob as the network input and run inference
        self.net.setInput(blob)
        detections = self.net.forward()       # shape: (1, 1, N, 7)

        boxes       = []
        confidences = []

        # ── Parse the raw SSD output tensor ─────────────────────────────
        # Each row: [batch_id, class_id, confidence, x1, y1, x2, y2]
        # Coordinates are normalised [0, 1] — multiply by (w, h) to get pixels
        for i in range(detections.shape[2]):
            row        = detections[0, 0, i]
            confidence = float(row[2])
            class_id   = int(row[1])

            # Keep only high-confidence "person" detections.
            # VOC class 15 = person (NOT 1 — that is COCO format).
            if class_id != PERSON_CLASS_ID or confidence < self.conf_threshold:
                continue

            x1 = max(0, int(row[3] * w))
            y1 = max(0, int(row[4] * h))
            x2 = min(w, int(row[5] * w))
            y2 = min(h, int(row[6] * h))

            boxes.append([x1, y1, x2 - x1, y2 - y1])   # NMS expects [x,y,w,h]
            confidences.append(confidence)

        if not boxes:
            return []

        # dnn – Non-Maximum Suppression removes redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold
        )

        results = []
        for idx in (indices.flatten() if len(indices) > 0 else []):
            x, y, bw, bh = boxes[idx]
            results.append({
                "box":        (x, y, x + bw, y + bh),
                "confidence": confidences[idx]
            })

        return results
