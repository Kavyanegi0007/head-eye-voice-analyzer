Overview

Tracks head pose and eye direction, logging the data and analyzing live voice from video or mic. Uses MediaPipe and other advanced models for improved accuracy.
Overall Flow
Loads sensitivity settings from config.json.


Initializes MediaPipe FaceMesh for face landmark detection.


Starts live video feed and optionally live audio capture.


Processes each video frame to:


Track eye direction (LEFT, RIGHT, CENTER, UP, DOWN).


Detect head direction (yaw/pitch-based).


Detect voice activity using RMS (root mean square) of live audio.


Log results per frame.

 Eye Direction Summary Report:
    LEFT:   92 frames ( 39.8%)
   RIGHT:    7 frames (  3.0%)
  CENTER:    9 frames (  3.9%)
      UP:   83 frames ( 35.9%)
    DOWN:   40 frames ( 17.3%)
  unknown:    0 frames (  0.0%)

 Head Direction Summary Report:
  Looking Left:    5 frames (  2.2%)
  Looking Right:    0 frames (  0.0%)
  Looking Down:    0 frames (  0.0%)
  Looking Up:   59 frames ( 25.5%)
  Forward:  167 frames ( 72.3%)

 Audio Detection Summary Report:
  Voice:    0 frames (  0.0%)
  Noise: 1761 frames (100.0%)

How CPU Usage Is Optimized
Frame Skipping: Only processes every 2nd frame, cutting CPU usage by ~50%
Downscaling the Frame: Reduces image size → faster processing by mediapipe and OpenCV.
Using Mediapipe Efficiently:  Processes only 1 face, not full scene.
Audio Thread Isolated: Runs voice_detection in a separate thread, ensuring it doesn’t block or slow down frame rendering.
Frame Sleep to Throttle Frame Rate: This is currently pausing for 2 seconds per frame
No Video Saving
Average CPU use: 11%
 Requirements:
Python 3.11
Packages:
pip install opencv-python mediapipe numpy sounddevice psutil
