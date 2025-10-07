import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from collections import deque
import threading
import time

# === CONFIG ===
FPS = 30
BUFFER_SIZE = 150  # ~5 seconds at 30 FPS
ROI_HISTORY = deque(maxlen=BUFFER_SIZE)
BPM_HISTORY = deque(maxlen=20)
PYRAMID_LEVEL = 4
ALPHA = 50.0
f_lo = 0.83  # Hz (~50 BPM)
f_hi = 3.0   # Hz (~180 BPM)
CHROMA_ATTENUATION = 0.1

# Face detector (load once)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Color Space ===
def rgb2yiq(rgb):
    M = np.array([[0.299, 0.587, 0.114],
                  [0.596, -0.274, -0.322],
                  [0.211, -0.523, 0.312]], dtype=np.float32)
    return np.tensordot(rgb, M.T, axes=([2], [1]))

def yiq2rgb(yiq):
    M_inv = np.array([[1.000,  0.956,  0.621],
                      [1.000, -0.272, -0.647],
                      [1.000, -1.106,  1.703]], dtype=np.float32)
    rgb = np.tensordot(yiq, M_inv.T, axes=([2], [1]))
    return np.clip(rgb, 0, 1)

# === Pyramid ===
def get_pyramid_level(img, level):
    for _ in range(level):
        img = cv2.pyrDown(img)
    return img

# === Temporal Filtering (FFT-based, applied on buffer) ===
def apply_temporal_filter(buffer, fps, f_lo, f_hi):
    N = len(buffer)
    if N < 30:  # need min data
        return np.zeros_like(buffer)
    freqs = np.fft.rfftfreq(N, d=1.0/fps)
    fft = np.fft.rfft(buffer, axis=0)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    fft_filtered = fft * mask[:, None]
    return np.fft.irfft(fft_filtered, n=N, axis=0).real

# === Plotting Thread ===
def update_plot():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    t_axis = np.arange(BUFFER_SIZE) / FPS
    freqs_bpm = np.fft.rfftfreq(BUFFER_SIZE, d=1.0/FPS) * 60

    while True:
        if len(ROI_HISTORY) < 30:
            time.sleep(0.1)
            continue

        signal_ts = np.array(ROI_HISTORY)
        if len(signal_ts) < BUFFER_SIZE:
            signal_ts = np.pad(signal_ts, (BUFFER_SIZE - len(signal_ts), 0), mode='edge')

        # Time domain
        ax1.clear()
        ax1.plot(t_axis, signal_ts, 'b-')
        ax1.set_title('Live Chrominance Signal (Forehead ROI)')
        ax1.set_ylabel('I/Q Value')
        ax1.set_ylim(signal_ts.min() - 0.01, signal_ts.max() + 0.01)
        ax1.grid(True)

        # Frequency domain (FFT)
        windowed = signal_ts * np.hamming(len(signal_ts))
        fft_mag = np.abs(np.fft.rfft(windowed))
        ax2.clear()
        ax2.plot(freqs_bpm, fft_mag, 'r-')
        ax2.set_xlim(40, 180)
        ax2.set_xlabel('Heart Rate (BPM)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Live FFT Spectrum')
        ax2.grid(True)

        # Show latest BPM if available
        if BPM_HISTORY:
            current_bpm = np.mean(BPM_HISTORY)
            ax2.axvline(current_bpm, color='g', linestyle='--', label=f'{current_bpm:.1f} BPM')
            ax2.legend()

        plt.tight_layout()
        plt.pause(0.01)
        time.sleep(0.2)

# === Main Real-Time Loop ===
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting real-time heart rate estimation...")
    print("Press 'q' to quit.")

    # Start plotting in background
    plot_thread = threading.Thread(target=update_plot, daemon=True)
    plot_thread.start()

    frame_count = 0
    pyramid_buffer = []  # only store current pyramid level per frame (not full frames)
    raw_frames_for_recon = []  # only needed if you want to show amplified video (optional)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) > 0:
            # Get largest face
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            # Forehead ROI
            rx, ry = x + int(0.3*w), y + int(0.1*h)
            rw, rh = int(0.4*w), int(0.2*h)

            # Extract ROI signal from original frame (no storage)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            yiq = rgb2yiq(rgb)
            roi_patch = yiq[ry:ry+rh, rx:rx+rw, 1]  # Use I channel (or 2 for Q)
            roi_mean = np.mean(roi_patch) if roi_patch.size > 0 else 0.0
            ROI_HISTORY.append(roi_mean)

            # Optional: draw ROI
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)

        # Estimate BPM every few frames
        if frame_count % 10 == 0 and len(ROI_HISTORY) >= 60:
            signal_ts = np.array(ROI_HISTORY)
            signal_ts = signal.detrend(signal_ts)
            N = len(signal_ts)
            freqs = np.fft.rfftfreq(N, d=1.0/FPS)
            fft_vals = np.abs(np.fft.rfft(signal_ts * np.hamming(N)))
            bpm_freqs = freqs * 60
            valid = (bpm_freqs >= 45) & (bpm_freqs <= 180)
            if np.any(valid):
                peak_idx = np.argmax(fft_vals[valid])
                true_idx = np.where(valid)[0][peak_idx]
                bpm = bpm_freqs[true_idx]
                BPM_HISTORY.append(bpm)
                cv2.putText(frame, f"BPM: {bpm:.1f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Real-Time Heart Rate', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()