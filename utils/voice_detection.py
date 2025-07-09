import numpy as np
import sounddevice as sd
import threading
from queue import Queue
import time
from utils.config_loader import load_config
config = load_config()


def record_audio(duration=0.2, samplerate=16000):
    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                       channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print(f"Audio capture failed: {e}")
        return np.array([])

def classify_audio(audio: np.ndarray, threshold=None):
    if threshold is None:
        from utils.config_loader import load_config
        config = load_config()
        threshold = config["audio_threshold"]

    if len(audio) == 0:
        return "noise"
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"[DEBUG] RMS: {rms:.6f}")
    return "voice" if rms > threshold else "noise"

def voice_detection_thread(voice_queue: Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        audio = record_audio()
        label = classify_audio(audio)
        voice_queue.put(label)


audio_log = []  # Add this at the top (shared global)

def voice_detection_thread(voice_queue: Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        audio = record_audio()
        label = classify_audio(audio)
        voice_queue.put(label)
        audio_log.append({
            "timestamp": time.time(),
            "label": label
        })
def print_audio_report(audio_log):
    total_entries = len(audio_log)
    if total_entries == 0:
        print("No audio data available.")
        return

    label_counts = {"voice": 0, "noise": 0}

    for entry in audio_log:
        label = entry.get("label", "noise")
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts["noise"] += 1  # default fallback

    print("\n🔊 Audio Detection Summary Report:")
    for label, count in label_counts.items():
        percentage = (count / total_entries) * 100
        print(f"  {label.capitalize():>5}: {count:>4} frames ({percentage:5.1f}%)")

    # Optional: return as dict for JSON saving
    return {
        "total": total_entries,
        "label_counts": label_counts
    }
