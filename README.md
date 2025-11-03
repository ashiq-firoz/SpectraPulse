# SpectraPulse - Real-time Heart Rate Detection

SpectraPulse is a computer vision-based application that estimates heart rate in real-time using facial video analysis. It utilizes the photoplethysmography (PPG) principle to detect subtle color changes in facial skin that correspond to blood flow variations.

## Features

- Real-time heart rate estimation
- Live signal visualization
- Face detection and ROI tracking
- Frequency domain analysis
- Interactive plots showing both time and frequency domain data

## Requirements

- Python 3.9+
- Webcam
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ashiq-firoz/SpectraPulse.git
cd SpectraPulse
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python t.py
```

Running the interface
```
streamlit run interface.py
```

- Position your face in front of the webcam
- The green rectangle shows the detected face
- The blue rectangle shows the forehead ROI used for measurement
- Press 'q' to quit the application

## How it Works

1. Face Detection: Uses Haar Cascade Classifier
2. ROI Extraction: Analyzes the forehead region
3. Signal Processing: Applies temporal filtering and FFT analysis
4. Visualization: Shows real-time plots of:
   - Raw signal in time domain
   - Frequency spectrum with BPM estimation

## Notes

- Maintain stable lighting conditions for better results
- Minimize head movement during measurement
- Allow a few seconds for the signal to stabilize


