# Video Anomaly Detection & Captioning

This project detects visual anomalies in videos using classic computer vision techniques (frame difference and SSIM) and generates captions for video chunks using BLIP. 
Outputs include a summary, frame grid, anomaly timeline, JSON report, and chunk analysis.

## Features
- Detects anomalies between video frames using frame difference and SSIM (Structural Similarity Index Measure).
- Generates captions for video chunks using BLIP (if available).
- Saves results as images, charts, and JSON reports.

## Requirements
- Python 3.8 or higher
- pip (Python package manager)

### Python Packages
Install the following packages:
- numpy
- opencv-python
- pillow
- matplotlib
- scikit-image
- transformers
- torch

You can install all required packages with:

```
pip install numpy opencv-python pillow matplotlib scikit-image transformers torch
```

## Setup
1. Place your input video at the path specified in `VIDEO_PATH` in `testing.py`.
   - Default: `C:\Users\Gustavo\Desktop\DrApurbasTasks\VideoAnomaly\VideoCaptioningEnglish\VideoAnomaly\input_video.mp4`
   - Change this path if needed.
2. Ensure you have all required packages installed.
3. (Optional) If you want to use BLIP captioning, make sure your system supports PyTorch and has enough memory.

## Running the Script
1. Open a terminal in the project directory.
2. Run the script:

```
python testing.py
```

## Output
- Results are saved in the `crash_output_real` folder:
  - `frames/`: All extracted frames
  - `collisions/`: Frames detected as anomalous
  - `report.json`: JSON summary of results
  - `anomaly_timeline.png`: Chart of frame differences over time
  - `all_frames_grid.jpg`: Grid of all frames

## Notes
- SSIM (Structural Similarity Index Measure) is used to compare how similar two images are. Lower SSIM means more difference.
- BLIP is used for generating captions and answering questions about video chunks. If BLIP is not available, only anomaly detection runs.
- The script prints progress and results in the terminal for easy tracking.

## Troubleshooting
- If you get errors about missing packages, install them with pip as shown above.
- If your video path is incorrect, update `VIDEO_PATH` in `testing.py`.
- For GPU acceleration, ensure PyTorch is installed with CUDA support.