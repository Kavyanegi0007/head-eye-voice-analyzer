import cv2
import threading
from queue import Queue
import mediapipe as mp
import numpy as np
import psutil
import time

# Your utility functions
from utils.head_pose import simple_head_pose_detection,print_head_pose_report, head_pose_log
from utils.eye_direction import simple_eye_direction
from utils.voice_detection import voice_detection_thread, audio_log
from utils.eye_direction import simple_eye_direction, print_eye_direction_report, eye_tracking_log
from utils.voice_detection import print_audio_report


def simple_test_main():
    # 🔁 Load video file (REPLACE with your video path)
    cap = cv2.VideoCapture(r"D:\live interview tracker\2\4\vid.mp4")
    if not cap.isOpened():
        print("❌ Cannot open video file")
        return

    # ✅ Start voice detection (live microphone)
    voice_queue = Queue()
    stop_event = threading.Event()
    voice_thread = threading.Thread(target=voice_detection_thread, args=(voice_queue, stop_event))
    voice_thread.start()

    frame_skip = 2
    frame_count = 0
    last_voice_label = "..."

    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ Video ended or frame read failed.")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # ↓ Resize frame to reduce CPU
            frame = cv2.resize(frame, (640, 480))

            # Detect head and eyes
            head_dir = simple_head_pose_detection(frame, face_mesh)
            
            eye_dir = simple_eye_direction(frame, face_mesh)
            

            # Get latest voice label (live)
            while not voice_queue.empty():
                last_voice_label = voice_queue.get()

            # Annotate video
            cv2.putText(frame, f"Head Dir: {head_dir}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Eye Dir: {eye_dir}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Voice: {last_voice_label}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # Display frame
            cv2.imshow("Main View", frame)
            print(f"CPU Usage: {psutil.cpu_percent()}%")

            # Frame rate control (optional)
            time.sleep(2)  # ~30 FPS

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ✅ Clean up
    stop_event.set()
    voice_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    eye_summary = print_eye_direction_report(eye_tracking_log)
    head_summary = print_head_pose_report(head_pose_log)
    print_audio_report(audio_log)
    import json
    summary_report = {
        "eye_direction_summary": eye_summary,
        "head_direction_summary": head_summary
    }

    with open("summary_report.json", "w") as f:
        json.dump(summary_report, f, indent=4)
    print("✅ Finished.")


if __name__ == "__main__":
    simple_test_main()
