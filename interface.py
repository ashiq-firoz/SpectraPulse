# streamlit_hr.py
import streamlit as st
import cv2
import numpy as np
from collections import deque
from scipy import signal
import matplotlib.pyplot as plt
import time

# --------------------
# === HYPERPARAMS ===
# --------------------
FPS = 30
BUFFER_SIZE = 150  # ~5 seconds @ 30 FPS
ROI_HISTORY = deque(maxlen=BUFFER_SIZE)
BPM_HISTORY = deque(maxlen=20)

# magnification / pyramid (kept for parity with original implementation)
ALPHA = 50.0
LEVEL = 4
SCALE_FACTOR = 1.0

# Desired heart-rate band 
f_lo = 50.0 / 60.0   # 0.83333 Hz (50 BPM)
f_hi = 60.0 / 60.0   # 1.0 Hz      (60 BPM)

# Face detector (load once)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --------------------
# === COLOR SPACE ===
# --------------------
def rgb2yiq(rgb):
    """Convert RGB image (float in [0,1]) to YIQ (returns float)"""
    # Using FCC-ish coefficients consistent with your provided code
    y = rgb @ np.array([[0.30], [0.59], [0.11]])
    rby = rgb[:, :, (0, 2)] - y
    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)
    yiq = np.dstack((y.squeeze(), i, q))
    return yiq

# --------------------
# === STREAMLIT UI ===
# --------------------
st.set_page_config(page_title="Real-time HR (Streamlit)", layout="wide")
st.title("💓 Real-time Heart Rate Estimation (Streamlit)")

col_left, col_center, col_right = st.columns([1, 1.2, 1])  # make center slightly bigger

with col_center:
    start_btn = st.button("▶ Start Webcam")
    stop_btn = st.button("⏹ Stop Webcam")

# placeholders
left_plot_ph = col_left.empty()
center_video_ph = col_center.empty()
bpm_text_ph = col_center.empty()
right_plot_ph = col_right.empty()
status_ph = st.empty()

# helper: draw matplotlib figures to placeholders
def plot_time_signal(ax, t_axis, signal_ts):
    ax.plot(t_axis, signal_ts)
    ax.set_title("Chrominance Signal (Forehead ROI)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("I-channel")
    ax.grid(True)

def plot_fft(ax, freqs_bpm, fft_mag, mean_bpm=None):
    ax.plot(freqs_bpm, fft_mag)
    ax.set_xlim(40, 180)
    ax.set_xlabel("Heart Rate (BPM)")
    ax.set_ylabel("Magnitude")
    ax.set_title("FFT Spectrum")
    if mean_bpm is not None:
        ax.axvline(mean_bpm, linestyle="--", label=f"{mean_bpm:.1f} BPM")
        ax.legend()
    ax.grid(True)

# main webcam loop (runs in response to button press)
def run_webcam_loop():
    status_ph.info("Starting webcam... ⏳")
    cap = cv2.VideoCapture(0)
    # Try to set FPS and resolution (driver dependent)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    start_time = time.time()

    # Precompute variables used for FFT axis
    t_axis_full = np.arange(BUFFER_SIZE) / FPS
    freqs_full = np.fft.rfftfreq(BUFFER_SIZE, d=1.0 / FPS)  # in Hz
    freqs_bpm_full = freqs_full * 60.0

    try:
        while cap.isOpened():
            # Allow Streamlit stop button to break loop
            if stop_btn:
                status_ph.info("Stopping (stop button pressed).")
                break

            ret, frame = cap.read()
            if not ret:
                status_ph.error("Can't read webcam frame.")
                break

            frame_disp = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                x, y, w, h = faces[0]
                # forehead ROI same as original
                rx, ry = x + int(0.3 * w), y + int(0.1 * h)
                rw, rh = int(0.4 * w), int(0.2 * h)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                yiq = rgb2yiq(rgb)
                roi_patch = yiq[ry:ry + rh, rx:rx + rw, 1]  # I channel
                roi_mean = np.mean(roi_patch) if roi_patch.size > 0 else 0.0
                ROI_HISTORY.append(roi_mean)

                # draw face and ROI boxes
                cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame_disp, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)

            # Estimate BPM every 10 frames (same logic as your original working code)
            if frame_count % 10 == 0 and len(ROI_HISTORY) >= 60:
                signal_ts = np.array(ROI_HISTORY)
                signal_ts = signal.detrend(signal_ts)  # remove trend
                N = len(signal_ts)

                # zero-pad or truncate to BUFFER_SIZE for consistent FFT axis
                if N < BUFFER_SIZE:
                    padded = np.pad(signal_ts, (BUFFER_SIZE - N, 0), mode='edge')
                else:
                    padded = signal_ts[-BUFFER_SIZE:]

                # windowing + FFT
                win = padded * np.hamming(len(padded))
                fft_vals = np.abs(np.fft.rfft(win))
                freqs = np.fft.rfftfreq(len(padded), d=1.0 / FPS)  # in Hz
                bpm_freqs = freqs * 60.0

                # Convert requested f_lo/f_hi (Hz) to BPM range mask
                bpm_lo = f_lo * 60.0
                bpm_hi = f_hi * 60.0
                valid_mask = (bpm_freqs >= bpm_lo) & (bpm_freqs <= bpm_hi)

                # avoid empty valid_mask
                if np.any(valid_mask):
                    # choose the peak inside the valid_mask
                    masked_fft = fft_vals.copy()
                    masked_fft[~valid_mask] = 0.0
                    peak_idx = np.argmax(masked_fft)
                    bpm = bpm_freqs[peak_idx]
                    if bpm > 0:
                        BPM_HISTORY.append(bpm)
                        # overlay BPM on frame
                        cv2.putText(frame_disp, f"BPM: {bpm:.1f}", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Show center video
            # convert BGR->RGB for Streamlit image
            frame_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            center_video_ph.image(frame_rgb, channels="RGB", use_column_width=True)

            # Show BPM text summary
            if BPM_HISTORY:
                mean_bpm = float(np.mean(BPM_HISTORY))
                bpm_text_ph.markdown(f"### 💓 Current: **{mean_bpm:.1f} BPM**")
            else:
                bpm_text_ph.markdown("### 💓 Current: **-- BPM**")

            # Left: time-domain signal (latest BUFFER_SIZE padded)
            if len(ROI_HISTORY) > 5:
                sig = np.array(ROI_HISTORY)
                if len(sig) < BUFFER_SIZE:
                    sig_p = np.pad(sig, (BUFFER_SIZE - len(sig), 0), mode="edge")
                else:
                    sig_p = sig[-BUFFER_SIZE:]
                fig1, ax1 = plt.subplots(figsize=(3.5, 2.5))
                plot_time_signal(ax1, np.arange(len(sig_p)) / FPS, sig_p)
                left_plot_ph.pyplot(fig1)
                plt.close(fig1)
            else:
                left_plot_ph.info("Waiting for ROI samples...")

            # Right: FFT plot (use padded window)
            if len(ROI_HISTORY) >= 30:
                # compute FFT using padded data (same as used for BPM detection above)
                padded = np.array(ROI_HISTORY)
                if len(padded) < BUFFER_SIZE:
                    padded = np.pad(padded, (BUFFER_SIZE - len(padded), 0), mode='edge')
                else:
                    padded = padded[-BUFFER_SIZE:]

                win = padded * np.hamming(len(padded))
                fft_vals = np.abs(np.fft.rfft(win))
                freqs = np.fft.rfftfreq(len(padded), d=1.0 / FPS)
                freqs_bpm = freqs * 60.0
                mean_bpm = np.mean(BPM_HISTORY) if BPM_HISTORY else None

                fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))
                plot_fft(ax2, freqs_bpm, fft_vals, mean_bpm)
                right_plot_ph.pyplot(fig2)
                plt.close(fig2)
            else:
                right_plot_ph.info("FFT will appear here after ~30 samples.")

            frame_count += 1
            # small sleep to yield control and keep close to target FPS
            key_wait = 1.0 / FPS
            time.sleep(key_wait)

    finally:
        cap.release()
        status_ph.success("Webcam stopped.")

# When Start clicked, run the webcam loop (with a spinner + message)
if start_btn:
    # clear previous histories so each session starts fresh
    ROI_HISTORY.clear()
    BPM_HISTORY.clear()
    # show immediate UI feedback
    status_ph.info("Starting webcam — please allow camera permission in your browser/OS.")
    # Run loop (blocking inside Streamlit; stops when stop_btn pressed)
    run_webcam_loop()
