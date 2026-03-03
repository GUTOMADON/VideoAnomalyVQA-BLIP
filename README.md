# Video Anomaly Detection & Captioning

Analyzes video footage to detect anomalies — crashes, fire, people down, wrong-way vehicles — using frame difference, SSIM, and BLIP with VQA.

---

## Install

```bash
pip install numpy opencv-python pillow matplotlib scikit-image transformers torch
```

---

## Usage

1. Place your video in the project folder and name it `input_video.mp4`  
   *(or change `VIDEO_PATH` inside the script)*

2. Run:
```bash
python video_anomaly_detection.py
```

---

## Output

All files are saved to `output_video/`:

| Path | Description |
|---|---|
| `frames/` | Every extracted frame |
| `collisions/` | Frames flagged as anomalous |
| `report.json` | Full per-frame results in JSON |
| `anomaly_timeline.png` | Frame difference chart over time |
| `all_frames_grid.jpg` | Visual grid of all frames |

---

