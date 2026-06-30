# Real-Time People Counter 🎯

A production-quality Python + OpenCV application that detects and counts people
in a live webcam feed using deep learning (MobileNet-SSD), while simultaneously
demonstrating **all eight** core OpenCV modules.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 👤 Person detection | MobileNet-SSD DNN (COCO-trained) |
| 🟩 Bounding boxes | Green box + confidence % per person |
| 🔢 People count | Live count in top-left HUD |
| 📊 Crowd status | Low / Medium / High crowd labels |
| 🏆 Session max | Tracks peak people count |
| ⚡ FPS display | Rolling average, top-right HUD |
| 📅 Date & time | Overlaid on every frame |
| 📐 Resolution | Width × Height shown in HUD |
| 🔵 ORB keypoints | Toggle with **O** key (features2d) |
| 😊 Face detection | Toggle with **F** key (Haar cascade) |
| 💾 Frame save | Press **S** — saved to `saved_frames/` |
| ❌ Quit | Press **Q** or close the window |

---

## 📁 Folder Structure

```
OpenCV/
├── main.py               # Entry point — orchestrates all modules
├── detector.py           # DNN person detection (MobileNet-SSD)
├── utils.py              # ORB, Haar, HUD drawing, save helpers
├── download_model.py     # One-click model downloader
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── models/               # Model weights (downloaded separately)
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
└── saved_frames/         # Auto-created when you press S
```

---

## 🔧 OpenCV Modules — Detailed Explanation

### 1. `core`
The foundation of every OpenCV operation.  
Used for: `np.ndarray` matrix operations, `cv2.addWeighted` (alpha blending
the HUD panel), ROI slice assignment for label backgrounds,
and `cv2.imencode` internally used by imgcodecs.

### 2. `videoio`
Handles all camera I/O.  
Used for: `cv2.VideoCapture(index)` to open the webcam,
`cap.read()` to grab frames, `cap.set()` to configure resolution/FPS/buffer,
`cap.release()` to safely free the device on exit.

### 3. `highgui`
The windowing and keyboard-event system.  
Used for: `cv2.namedWindow`, `cv2.imshow`, `cv2.waitKey` (1 ms keyboard poll),
`cv2.destroyAllWindows`, `cv2.getWindowProperty` (detect window close).

### 4. `imgproc`
Image processing and drawing primitives.  
Used for: `cv2.cvtColor` (BGR→Gray for ORB and Haar),
`cv2.equalizeHist` (improve face detection in poor lighting),
`cv2.GaussianBlur` (denoise before DNN), `cv2.resize` (downscale for speed),
`cv2.rectangle`, `cv2.putText`, `cv2.getTextSize`, `cv2.LINE_AA` (anti-alias).

### 5. `imgcodecs`
Image file encoding and decoding.  
Used for: `cv2.imwrite` in `save_frame()` — encodes the frame as JPEG (quality 95)
and writes it to `saved_frames/` when the user presses **S**.

### 6. `features2d`
2D feature detection and description.  
Used for: `cv2.ORB_create()` — an ORB detector that finds up to 300
rotation-invariant binary keypoints per frame. `cv2.drawKeypoints` with
`DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` renders coloured circles scaled
to each keypoint's response strength. Toggled with **O**.

### 7. `objdetect`
Classical object detection via cascade classifiers.  
Used for: `cv2.CascadeClassifier` loaded with the built-in
`haarcascade_frontalface_default.xml`. `detectMultiScale` scans the
equalised greyscale frame for frontal faces and returns bounding boxes.
Results are drawn as cyan-gold rectangles. Toggled with **F**.

### 8. `dnn`
Deep neural network inference engine.  
Used for: `cv2.dnn.readNetFromCaffe` — loads MobileNet-SSD architecture and
weights. `cv2.dnn.blobFromImage` — normalises and reshapes the frame into a
4-D NCHW tensor. `net.forward()` — runs the SSD inference pass.
`cv2.dnn.NMSBoxes` — suppresses duplicate detections via IoU thresholding.

---

## 🚀 Installation & Quick Start

### Prerequisites
- Python 3.10 or higher
- A working webcam
- Internet connection (for downloading the model)

### Step 1 — Clone / open the project

```bash
cd c:\OpenCV
```

### Step 2 — Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> `opencv-contrib-python` includes all extra modules (ORB, CUDA, etc.).

### Step 4 — Download the MobileNet-SSD model

```bash
python download_model.py
```

This downloads two files into `models/`:

| File | Size |
|---|---|
| `MobileNetSSD_deploy.prototxt` | ~31 KB |
| `MobileNetSSD_deploy.caffemodel` | ~23 MB |

**Manual download** (if the script fails):  
- Prototxt: https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt  
- Caffemodel: https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel  

Place both files inside the `models/` directory.

### Step 5 — Run the application

```bash
python main.py
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `S` | Save current frame to `saved_frames/` |
| `O` | Toggle ORB keypoint overlay |
| `F` | Toggle Haar face detection overlay |

---

## 🎛️ Configuration

Edit the constants at the top of `main.py`:

```python
CAMERA_INDEX   = 0      # Change to 1, 2 … for other cameras
FRAME_WIDTH    = 1280   # Capture resolution
FRAME_HEIGHT   = 720
CONF_THRESHOLD = 0.45   # Lower → more detections (less accurate)
NMS_THRESHOLD  = 0.40   # Lower → fewer overlapping boxes kept
DETECT_EVERY_N = 2      # Run DNN every N frames (higher = faster)
```

---

## 📊 Crowd Status Thresholds

| People count | Status | Colour |
|---|---|---|
| 0 – 3 | LOW CROWD | 🟢 Green |
| 4 – 8 | MEDIUM CROWD | 🟠 Orange |
| 9+ | HIGH CROWD | 🔴 Red |

---

## ⚡ Performance Tips

- Set `DETECT_EVERY_N = 3` or higher on slow machines.
- Set `use_gpu=True` in `PersonDetector(...)` if you have CUDA-capable GPU
  and installed `opencv-contrib-python` with CUDA support.
- Reduce `FRAME_WIDTH / FRAME_HEIGHT` for higher FPS on low-end hardware.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open camera` | Check another app isn't using the webcam; try `CAMERA_INDEX = 1` |
| `FileNotFoundError: models/…` | Run `python download_model.py` first |
| `Failed to load Haar cascade` | Ensure `opencv-contrib-python` (not just `opencv-python`) is installed |
| Low FPS | Increase `DETECT_EVERY_N`, reduce resolution, or enable GPU |
| Black / frozen window | Confirm webcam drivers are up to date |

---

## 📦 Requirements

```
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy>=1.24.0
requests>=2.28.0
tqdm>=4.65.0
```

---

## 📄 License

This project is provided for educational purposes under the MIT License.
