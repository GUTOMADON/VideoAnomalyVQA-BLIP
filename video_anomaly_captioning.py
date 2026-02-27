
"""
Video Anomaly Detection & Captioning
Detects visual anomalies in video using classic CV 
(frame difference + SSIM) and generates chunk captions with BLIP. 
Outputs: summary, grid, timeline, JSON, chunk analysis.

SSIM: Measures how similar two images are (higher = more similar).
"""

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
except ImportError:

    BlipProcessor = BlipForConditionalGeneration = torch = None

 # BLIP utility functions
def load_blip():
    if not (BlipProcessor and BlipForConditionalGeneration and torch):
        return None, None, None
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

def describe_frame(img, processor, model, device):

    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    return processor.decode(out[0], skip_special_tokens=True)

 # BLIP VQA utility function
def vqa_frame(img, processor, model, device, question):
    # Use BLIP for VQA if supported
    try:
        inputs = processor(images=img, text=question, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=10)
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer
    except Exception:
        return None

import os
import sys
import json
import math
import shutil
import stat
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
from datetime import datetime
from pathlib import Path


 # Configuration
OUTPUT_DIR      = "output_video"
FRAMES_DIR      = os.path.join(OUTPUT_DIR, "frames")
ANOMALY_DIR     = os.path.join(OUTPUT_DIR, "collisions")
REPORT_PATH     = os.path.join(OUTPUT_DIR, "report.json")
CHART_PATH      = os.path.join(OUTPUT_DIR, "anomaly_timeline.png")
GRID_PATH       = os.path.join(OUTPUT_DIR, "all_frames_grid.jpg")


VIDEO_PATH      = r"C:\Users\Gustavo\Desktop\DrApurbasTasks\VideoAnomaly\VideoCaptioningEnglish\VideoAnomaly\input_video.mp4"
EXTRACT_FPS     = 1.0   # Frames per second to extract
MAX_FRAMES      = 30    # Maximum frames to process


 # Utility functions
def clean_output_dirs():

    for d in (FRAMES_DIR, ANOMALY_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

def get_video_path():

    return VIDEO_PATH

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return []
    frames, frame_idx, sec = [], 0, 0.0
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret or len(frames) >= MAX_FRAMES:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append({
            "frame_idx": frame_idx,
            "time_sec": sec,
            "image": Image.fromarray(img),
        })
        frame_idx += 1
        sec += 1.0 / EXTRACT_FPS
    cap.release()
    return frames
 # Main pipeline
def run():

    print("=" * 60)
    print("VIDEO ANOMALY DETECTION & CAPTIONING")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("-" * 60)

    print("[SETUP]")
    clean_output_dirs()
    print("Output folders cleaned.")
    print("-" * 60)

    print("[VIDEO FRAME EXTRACTION]")
    video_path = get_video_path()
    frames = extract_frames(video_path)
    if not frames:
        print("ERROR: No frames extracted.")
        sys.exit(1)
    print(f"Total frames extracted: {len(frames)}")
    print("-" * 60)

    print("[ANOMALY DETECTION]")
    DIFF_SIZE = (96, 96)
    diffs, ssims = [0.0], [1.0]
    prev = np.array(frames[0]["image"].resize(DIFF_SIZE)).astype(np.float32)
    prev_gray = cv2.cvtColor(np.uint8(prev), cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        curr = np.array(frames[i]["image"].resize(DIFF_SIZE)).astype(np.float32)
        curr_gray = cv2.cvtColor(np.uint8(curr), cv2.COLOR_RGB2GRAY)
        diffs.append(np.mean(np.abs(curr - prev)))
        ssims.append(ssim(curr_gray, prev_gray, data_range=255))
        prev, prev_gray = curr, curr_gray
    mean_diff, std_diff = np.mean(diffs), np.std(diffs)
    mean_ssim, std_ssim = np.mean(ssims), np.std(ssims)
    threshold_diff = mean_diff + 2 * std_diff
    threshold_ssim = mean_ssim - 2 * std_ssim
    anom_active = False
    low_diff = threshold_diff * 0.5
    high_ssim = threshold_ssim * 1.5
    frame_results = []
    for i, fr in enumerate(frames):
        diff, ssim_val = diffs[i], ssims[i]
        is_anom_now = (diff > threshold_diff) or (ssim_val < threshold_ssim)
        if is_anom_now:
            anom_active = True
        elif diff < low_diff and ssim_val > high_ssim:
            anom_active = False
        is_anom = anom_active
        status = "ANOMALY" if is_anom else "Normal"
        print(f"Frame {fr['frame_idx']:>3}: Time={fr['time_sec']:>5.1f}s | Diff={diff:>7.3f} | SSIM={ssim_val:>6.3f} | Status: {status}")
        fname = f"frame_{fr['frame_idx']:05d}_t{fr['time_sec']:.2f}s.jpg"
        fr["image"].save(os.path.join(FRAMES_DIR, fname), quality=90)
        if is_anom:
            fr["image"].save(os.path.join(ANOMALY_DIR, fname), quality=90)
        frame_results.append({
            "frame_idx": int(fr["frame_idx"]),
            "time_sec": float(fr["time_sec"]),
            "difference": float(diff),
            "ssim": float(ssim_val),
            "is_anomaly": bool(is_anom),
        })
    print(f"Diff threshold: {threshold_diff:.3f} | SSIM threshold: {threshold_ssim:.3f}")
    print("-" * 60)

    processor, model, device = load_blip()
    chunk_descriptions, chunk_violations = [], []
    if processor:
        print("[BLIP CAPTIONING & QA]")
        chunk_size = 10
        chunks = [frames[i:i+chunk_size] for i in range(0, len(frames), chunk_size)]
        vqa_question = "Is there any traffic violation, accident, or collision in this scene? Answer yes or no."
        for idx, chunk in enumerate(chunks):
            print(f"Chunk {idx+1}/{len(chunks)}:")
            mid_frame = chunk[len(chunk)//2]["image"]
            desc = describe_frame(mid_frame, processor, model, device)
            chunk_descriptions.append(desc)
            vqa_answer = vqa_frame(mid_frame, processor, model, device, vqa_question)
            if vqa_answer is not None and vqa_answer.strip().lower() in ["yes", "no"]:
                found = (vqa_answer.strip().lower() == "yes")
                vqa_used = True
            else:
                found = any(word in desc.lower() for word in ["crash", "collision", "accident", "violation", "damage", "break", "hit"])
                vqa_used = False
            chunk_violations.append(found)
            status = "VIOLATION" if found else "Normal"
            vqa_note = f" [VQA: {vqa_answer}]" if vqa_used else " [KW]"
            print(f"Status: {status} | Description: {desc}{vqa_note}")
    else:
        print("[BLIP CAPTIONING & QA] BLIP not available. Only anomaly detection will run.")
    print("-" * 60)

    print("[CHUNK QA SUMMARY]")
    any_violation = any(chunk_violations)
    print(f"Any violation detected? {'Yes' if any_violation else 'No'}")
    violation_chunks = [i+1 for i, v in enumerate(chunk_violations) if v]
    if violation_chunks:
        print(f"Chunks with violation: {', '.join(map(str, violation_chunks))}")
    else:
        print("No violations detected in any chunk.")
    n_viol = sum(1 for v in chunk_violations if v)
    n_norm = sum(1 for v in chunk_violations if not v)
    print(f"Total chunks — Violation: {n_viol} | Normal: {n_norm}")
    print("-" * 60)

    print("[OUTPUT SAVING]")
    # Save grid, timeline, JSON, summary

    # Step 0: Clean previous output
    print("[Step 0] Cleaning previous output...")
    clean_output_dirs()

    # Step 1: Get video
    print("[Step 1] Obtaining video...")
    video_path = get_video_path()
    print()

    # Step 2: Extract frames from video
    print("[Step 2] Extracting frames...")
    frames = extract_frames(video_path)
    if not frames:
        print("ERROR: No frames were extracted.")
        sys.exit(1)
    print()

    # Step 2.5: Chunking and description (after anomaly detection)
    chunk_descriptions = []
    chunk_violations = []

    # Step 0: Clean previous output
    print("[Step 0] Cleaning previous output...")
    clean_output_dirs()

    # Step 1: Get video
    print("[Step 1] Obtaining video...")
    video_path = get_video_path()
    print()

    # Step 2: Extract frames from video
    print("[Step 2] Extracting frames...")
    frames = extract_frames(video_path)
    if not frames:
        print("ERROR: No frames were extracted.")
        sys.exit(1)
    print()

    # Step 3: Anomaly detection (frame difference)
    print("\n=====================[ STEP 3: ANOMALY DETECTION ]=====================")
    print("Detecting anomalies by frame difference\n")
    header = f"{'#':<3} | {'Frame':<7} | {'Time (s)':>8} | {'Diff':>8} | {'Status':^9} | Description"
    print(header)
    print("-" * len(header))

    # Use smaller resize for more sensitivity
    DIFF_SIZE = (96, 96)
    diffs = [0.0]  # First frame has no previous
    ssims = [1.0]  # First frame is identical to itself
    prev = np.array(frames[0]["image"].resize(DIFF_SIZE)).astype(np.float32)
    prev_gray = cv2.cvtColor(np.uint8(prev), cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        curr = np.array(frames[i]["image"].resize(DIFF_SIZE)).astype(np.float32)
        curr_gray = cv2.cvtColor(np.uint8(curr), cv2.COLOR_RGB2GRAY)
        diff = np.mean(np.abs(curr - prev))
        diffs.append(diff)
        ssim_val = ssim(curr_gray, prev_gray, data_range=255)
        ssims.append(ssim_val)
        prev = curr
        prev_gray = curr_gray

    # Thresholds: mean + 2*std for diff, mean - 2*std for SSIM
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    threshold_diff = mean_diff + 2 * std_diff
    mean_ssim = np.mean(ssims)
    std_ssim = np.std(ssims)
    threshold_ssim = mean_ssim - 2 * std_ssim
    print(f"\n[DEBUG] All frame differences: {diffs}")
    print(f"[DEBUG] All SSIM values: {ssims}")
    print(f"[DEBUG] Diff threshold: {threshold_diff:.4f} (mean={mean_diff:.4f}, std={std_diff:.4f})")
    print(f"[DEBUG] SSIM threshold: {threshold_ssim:.4f} (mean={mean_ssim:.4f}, std={std_ssim:.4f})\n")


    # Hysteresis: keep anomaly status until diff drops well below threshold
    anom_active = False
    low_threshold_diff = threshold_diff * 0.5
    high_threshold_ssim = threshold_ssim * 1.5

    frame_results = []
    for i, fr in enumerate(frames):
        diff = diffs[i]
        ssim_val = ssims[i]
        # Anomaly if diff is high OR ssim is low
        is_anom_now = (diff > threshold_diff) or (ssim_val < threshold_ssim)
        # Hysteresis: keep anomaly status until both metrics are back to normal
        if is_anom_now:
            anom_active = True
        elif diff < low_threshold_diff and ssim_val > high_threshold_ssim:
            anom_active = False
        is_anom = anom_active

        if is_anom:
            status_colored = "\033[91mANOMALY\033[0m"
        else:
            status_colored = "\033[92mNormal\033[0m"
        print(f"{i+1:<3} | {fr['frame_idx']:<7} | {fr['time_sec']:>8.2f} | {diff:>8.4f} | {ssim_val:>6.3f} | {status_colored:^9}")

        # Save frame images
        fname = f"frame_{fr['frame_idx']:05d}_t{fr['time_sec']:.2f}s.jpg"
        out_path = os.path.join(FRAMES_DIR, fname)
        fr["image"].save(out_path, quality=90)
        if is_anom:
            anom_path = os.path.join(ANOMALY_DIR, fname)
            fr["image"].save(anom_path, quality=90)

        frame_results.append({
            "frame_idx": int(fr["frame_idx"]),
            "time_sec": float(fr["time_sec"]),
            "difference": float(diff),
            "ssim": float(ssim_val),
            "is_anomaly": bool(is_anom),
        })

    print("\n" + "="*68)
    print(f"Diff threshold: {threshold_diff:.4f} (mean + 2*std)")
    print(f"SSIM threshold: {threshold_ssim:.4f} (mean - 2*std)")
    print(f"All frames saved in: {FRAMES_DIR}")
    print(f"Anomalous frames also saved in: {ANOMALY_DIR}")
    print("="*68 + "\n")

    # ... code...
    # Step 3.5: BLIP chunking and VQA/question answering (after anomaly detection)
    if processor:
        print("\n[Step 3.5] Generating descriptions and VQA for video chunks...\n")
        chunk_size = 10
        chunks = [frames[i:i+chunk_size] for i in range(0, len(frames), chunk_size)]
        keywords = ["crash", "collision", "accident", "violation", "damage", "break", "hit"]
        vqa_question = "Is there any traffic violation, accident, or collision in this scene? Answer yes or no."
        for idx, chunk in enumerate(chunks):
            print(f"\n[BLIP] Processing chunk {idx+1}/{len(chunks)}...")
            mid_frame = chunk[len(chunk)//2]["image"]
            desc = describe_frame(mid_frame, processor, model, device)
            chunk_descriptions.append(desc)
            # Try VQA first
            vqa_answer = vqa_frame(mid_frame, processor, model, device, vqa_question)
            if vqa_answer is not None and vqa_answer.strip().lower() in ["yes", "no"]:
                found = (vqa_answer.strip().lower() == "yes")
                vqa_used = True
            else:
                # Fallback to keyword search
                found = any(word in desc.lower() for word in keywords)
                vqa_used = False
            chunk_violations.append(found)
            color = "\033[91m" if found else "\033[92m"
            status = "VIOLATION" if found else "Normal"
            vqa_note = f" [VQA: {vqa_answer}]" if vqa_used else " [KW]"
            print(f"Chunk {idx+1}: {color}{status}\033[0m — {desc}{vqa_note}\n")
    else:
        print("BLIP not available. Only classic anomaly detection will run.")
        chunk_violations = []
        chunk_descriptions = []

    # Chunk QA logic (after anomaly detection)
    print("\n[Step 6.5] Question Answering on Chunks and Anomalies:\n")
    # Print chunk violations and descriptions
    print("[INFO] BLIP Chunk Analysis Summary (only frames analyzed by BLIP):")
    print("-" * 60)
    if chunk_violations and len(chunk_violations) == len(chunk_descriptions):
        any_violation = any(chunk_violations)
        print(f"1. Violation detected? {'Yes' if any_violation else 'No'}")
        violation_chunks = [i+1 for i, v in enumerate(chunk_violations) if v]
        if violation_chunks:
            print(f"2. Chunks with violation: {', '.join(map(str, violation_chunks))}")
        else:
            print("2. No violations detected in any chunk.")
        n_viol = sum(1 for v in chunk_violations if v)
        n_norm = sum(1 for v in chunk_violations if not v)
        print(f"3. Chunks — Violation: {n_viol} | Normal: {n_norm}")
        chunk_size = 10
        chunk_anomaly_overlap = []
        for idx, is_viol in enumerate(chunk_violations):
            if is_viol:
                start = idx * chunk_size
                end = min(start + chunk_size, len(frame_results))
                if any(fr['is_anomaly'] for fr in frame_results[start:end]):
                    chunk_anomaly_overlap.append(idx+1)
        if chunk_anomaly_overlap:
            print(f"4. Chunks with both violation and anomaly: {', '.join(map(str, chunk_anomaly_overlap))}")
        else:
            print("4. No chunk had both a violation and an anomaly.")
        print("-" * 60)
        print("\nDetailed BLIP Chunk Results:")
        print(f"{'Chunk':<6} | {'Status':<10} | {'Description'}")
        print("-" * 60)
        for idx, (desc, viol) in enumerate(zip(chunk_descriptions, chunk_violations)):
            status = "VIOLATION" if viol else "Normal"
            print(f"{idx+1:<6} | {status:<10} | {desc}")
        print("-" * 60)
    else:
        print("No chunk-based QA available (BLIP not run or no chunks).")

    # Save all frames grid
    print("[Step 4] Saving all frames grid...")
    N_COLS = 6
    THUMB_W = 210
    THUMB_H = 130
    LABEL_H = 52
    CELL_H = THUMB_H + LABEL_H
    n_rows = math.ceil(len(frame_results) / N_COLS)
    grid = Image.new("RGB", (N_COLS * THUMB_W, n_rows * CELL_H), (14, 14, 26))
    draw = ImageDraw.Draw(grid)
    for pos, fr in enumerate(frame_results):
        col, row = pos % N_COLS, pos // N_COLS
        x, y = col * THUMB_W, row * CELL_H
        is_an = fr["is_anomaly"]
        # Use the original frames list to get the image
        thumb = frames[pos]["image"].resize((THUMB_W, THUMB_H), Image.LANCZOS)
        grid.paste(thumb, (x, y))
        bg = (145, 15, 15) if is_an else (15, 95, 25)
        draw.rectangle([(x, y + THUMB_H), (x + THUMB_W - 1, y + CELL_H - 1)], fill=bg)
        status = "ANOMALY" if is_an else "Normal"
        draw.text((x + 4, y + THUMB_H + 3), status, fill=(255, 80, 80) if is_an else (80, 255, 100))
        draw.text((x + 4, y + THUMB_H + 18), f"t={fr['time_sec']:.1f}s  frm={fr['frame_idx']}", fill=(200, 200, 200))
        border_col = (210, 25, 25) if is_an else (25, 155, 55)
        draw.rectangle([(x, y), (x + THUMB_W - 1, y + CELL_H - 1)], outline=border_col, width=3)
    grid.save(GRID_PATH, quality=90)
    print(f"All frames grid saved: {GRID_PATH}")


    # Save anomaly timeline chart
    print("[Step 5] Saving anomaly timeline chart...")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    times = [fr["time_sec"] for fr in frame_results]
    diffs_plot = [fr["difference"] for fr in frame_results]
    ax.plot(times, diffs_plot, color="#4fc3f7", label="Frame difference", zorder=2)
    ax.axhline(threshold_diff, color="#ffd700", linestyle="--", linewidth=2, zorder=4, label="Anomaly threshold")
    for i, fr in enumerate(frame_results):
        if fr["is_anomaly"]:
            ax.scatter(fr["time_sec"], fr["difference"], color="#e72f2f", s=60, zorder=5)
    ax.set_title("Anomaly Detection — Frame Difference Over Time", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Time (seconds)", color="white", fontsize=11)
    ax.set_ylabel("Frame difference", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#ffffff22")
    ax.grid(axis="y", color="#ffffff1a", linestyle="--", lw=0.8, zorder=0)
    handles = [
        mpatches.Patch(color="#4fc3f7", label="Normal frame"),
        mpatches.Patch(color="#e84545", label="Anomalous frame"),
        plt.Line2D([0], [0], color="#ffd700", linestyle="--", label="Threshold"),
    ]
    ax.legend(handles=handles, facecolor="#0f0f1a", labelcolor="white", edgecolor="#ffffff33", fontsize=9)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Anomaly timeline chart saved: {CHART_PATH}")

    # Save the final report as JSON
    print("[Step 6] Saving final report as JSON...")
    def to_float(val):
        # Convert numpy float32/float64 to Python float
        if isinstance(val, (np.floating,)):
            return float(val)
        return val

    report = {
        "created_at": datetime.now().isoformat(),
        "video_path": video_path,
        "extract_fps": float(EXTRACT_FPS),
        "max_frames": int(MAX_FRAMES),
        "threshold_diff": to_float(threshold_diff),
        "threshold_ssim": to_float(threshold_ssim),
        "frames": [
            {
                "frame_idx": int(fr["frame_idx"]),
                "time_sec": to_float(fr["time_sec"]),
                "difference": to_float(fr["difference"]),
                "ssim": to_float(fr["ssim"]),
                "is_anomaly": bool(fr["is_anomaly"]),
            }
            for fr in frame_results
        ],
        "summary": {
            "total_frames": int(len(frame_results)),
            "anomalous_frames": int(sum(1 for fr in frame_results if fr["is_anomaly"])),
            "first_anomaly_time": to_float(next((fr["time_sec"] for fr in frame_results if fr["is_anomaly"]), None)),
            "last_anomaly_time": to_float(next((fr["time_sec"] for fr in reversed(frame_results) if fr["is_anomaly"]), None)),
        }
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Final report saved: {REPORT_PATH}")

    # Step 7: Print summary of results
    print("\n[Step 7] Summary of Results")
    total_frames = len(frame_results)
    anomaly_count = sum(1 for fr in frame_results if fr["is_anomaly"])
    normal_count = total_frames - anomaly_count
    print(f"Frames captured: \033[93m{total_frames}\033[0m")
    print(f"Anomalous frames: \033[91m{anomaly_count}\033[0m")
    print(f"Normal frames: \033[92m{normal_count}\033[0m")
    print("==========================\n")

if __name__ == "__main__":
    run()