# Head-Eye-Voice Analyzer

Tracks **head pose**, **eye direction**, and **live voice activity** from video and mic input using **MediaPipe**, **OpenCV**, and audio analysis. Outputs frame-wise logs and generates summarized reports.

---

## 🔄 Overview / Flow

1. Loads sensitivity settings from `config.json`.
2. Initializes MediaPipe FaceMesh for facial landmark detection.
3. Starts live video feed and (optionally) microphone input.
4. Processes each video frame to:
   - Track **eye direction**: `LEFT`, `RIGHT`, `CENTER`, `UP`, `DOWN`
   - Detect **head direction**: based on yaw/pitch
   - Detect **voice activity**: using RMS of live audio
5. Logs per-frame results
6. Generates a final **JSON + console report**

---

## 📊 Sample Summary Reports

### 👁️ Eye Direction Summary
LEFT : 92 frames (39.8%)
RIGHT : 7 frames ( 3.0%)
CENTER : 9 frames ( 3.9%)
UP : 83 frames (35.9%)
DOWN : 40 frames (17.3%)
unknown: 0 frames ( 0.0%)


### 🧠 Head Direction Summary
Looking Left : 5 frames ( 2.2%)
Looking Right: 0 frames ( 0.0%)
Looking Down : 0 frames ( 0.0%)
Looking Up : 59 frames (25.5%)
Forward : 167 frames (72.3%)


### 🎙️ Audio Detection Summary
Voice: 0 frames ( 0.0%)
Noise: 1761 frames (100.0%)


---

## ⚙️ CPU Optimization

- **✅ Frame Skipping**: Only processes every 2nd frame → ~50% CPU cut
- **✅ Downscaled Frames**: Resize to 640x480 before MediaPipe
- **✅ Efficient MediaPipe Use**: Processes only 1 face
- **✅ Audio Detection in Background Thread**
- **✅ Frame Rate Throttle**: 2-second pause per frame (can be adjusted)
- **❌ No video saving** to disk (saves CPU)
- **💻 Average CPU Use**: ~11% (depending on system)

---

## 🛠️ Requirements

- Python 3.11+
- Install dependencies:

```bash
pip install opencv-python mediapipe numpy sounddevice psutil

head-eye-voice-analyzer/
├── main.py
├── utils/
│   ├── eye_direction.py
│   ├── head_pose.py
│   ├── voice_detection.py
│   └── config_loader.py
├── config.json
├── summary_report.json
├── README.md
└── .gitignore
