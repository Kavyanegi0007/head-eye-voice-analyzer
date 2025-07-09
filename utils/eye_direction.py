import cv2
import numpy as np
import mediapipe as mp
from utils.config_loader import load_config
config = load_config()

face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # CRITICAL for iris detection
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

eye_tracking_log = []

def simple_eye_direction(frame, face_mesh ,timestamp=None , eye_tracking_log=eye_tracking_log):

    img_h, img_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return "unknown"
    
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Eye region landmarks (more comprehensive)
    LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # Eye corners for reference
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    
    def get_landmark_point(idx):
        return np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h])
    
    def find_pupil_in_eye_region(eye_landmarks):
        # Get eye region bounds
        eye_points = [get_landmark_point(i) for i in eye_landmarks]
        eye_points = np.array(eye_points, dtype=np.int32)
        
        # Create bounding box around eye
        x_min, y_min = np.min(eye_points, axis=0)
        x_max, y_max = np.max(eye_points, axis=0)
        
        # Add padding
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_w, x_max + padding)
        y_max = min(img_h, y_max + padding)
        
        # Extract eye region
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        if eye_region.size == 0:
            return None
        
        # Convert to grayscale for pupil detection
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_eye, (5, 5), 0)
        
        # Find the darkest point (pupil)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        # Convert back to full image coordinates
        pupil_x = x_min + min_loc[0]
        pupil_y = y_min + min_loc[1]
        
        # Draw eye region for debugging
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
        
        return np.array([pupil_x, pupil_y])
    
    # Find pupils
    left_pupil = find_pupil_in_eye_region(LEFT_EYE_LANDMARKS)
    right_pupil = find_pupil_in_eye_region(RIGHT_EYE_LANDMARKS)
    
    if left_pupil is None or right_pupil is None:
        return "unknown"
    
    # Get eye corners
    left_inner = get_landmark_point(LEFT_EYE_INNER)
    left_outer = get_landmark_point(LEFT_EYE_OUTER)
    right_inner = get_landmark_point(RIGHT_EYE_INNER)
    right_outer = get_landmark_point(RIGHT_EYE_OUTER)
    # Get top and bottom points for vertical ratio
    left_top = get_landmark_point(LEFT_EYE_TOP)
    left_bottom = get_landmark_point(LEFT_EYE_BOTTOM)
    right_top = get_landmark_point(RIGHT_EYE_TOP)
    right_bottom = get_landmark_point(RIGHT_EYE_BOTTOM)

    
    # Calculate horizontal ratios
    # def get_horizontal_ratio(pupil, inner, outer):
    #     eye_width = abs(outer[0] - inner[0])
    #     if eye_width == 0:
    #         return 0.5
        
    #     # Distance from inner corner to pupil
    #     pupil_offset = abs(pupil[0] - inner[0])
    #     ratio = pupil_offset / eye_width
        
    #     return max(0, min(1, ratio))
    def get_horizontal_ratio(pupil, inner, outer):
        eye_width = abs(outer[0] - inner[0])
        if eye_width == 0:
            return 0.5
        pupil_offset = abs(pupil[0] - inner[0])
        ratio = pupil_offset / eye_width
        return max(0, min(1, ratio))

    def get_vertical_ratio(pupil, top, bottom):
        eye_height = abs(bottom[1] - top[1])
        if eye_height == 0:
            return 0.5
        pupil_offset = abs(pupil[1] - top[1])
        ratio = pupil_offset / eye_height
        return max(0, min(1, ratio))
    
    left_ratio = get_horizontal_ratio(left_pupil, left_inner, left_outer)
    right_ratio = get_horizontal_ratio(right_pupil, right_inner, right_outer)
    
    avg_ratio = (left_ratio + right_ratio) / 2
    left_vertical_ratio = get_vertical_ratio(left_pupil, left_top, left_bottom)
    right_vertical_ratio = get_vertical_ratio(right_pupil, right_top, right_bottom)
    avg_vertical_ratio = (left_vertical_ratio + right_vertical_ratio) / 2

    
    # Debug information
    #print(f"Left pupil: {left_pupil}, Right pupil: {right_pupil}")
    #print(f"Left ratio: {left_ratio:.3f}, Right ratio: {right_ratio:.3f}, Avg: {avg_ratio:.3f}")
    # direction = "unknown"
    # if avg_ratio < config["eye_ratio_left_threshold"]:
    #     direction = "LEFT"

    
    # elif avg_ratio > config["eye_ratio_right_threshold"]:
    #     direction = "RIGHT"
    
    # else:
    #     direction = "CENTER"
    direction = "unknown"
    if avg_vertical_ratio < config["eye_ratio_up_threshold"]:
        direction = "UP"
    elif avg_vertical_ratio > config["eye_ratio_down_threshold"]:
        direction = "DOWN"
    elif avg_ratio < config["eye_ratio_left_threshold"]:
        direction = "LEFT"
    elif avg_ratio > config["eye_ratio_right_threshold"]:
        direction = "RIGHT"
    else:
        direction = "CENTER"

    eye_tracking_log.append({
        "timestamp": timestamp,
        "left_pupil": left_pupil.tolist(),
        "right_pupil": right_pupil.tolist(),
        "left_ratio": round(left_ratio, 3),
        "right_ratio": round(right_ratio, 3),
        "avg_ratio": round(avg_ratio, 3),
        "direction": direction,
        "left_vertical_ratio": round(left_vertical_ratio, 3),
        "right_vertical_ratio": round(right_vertical_ratio, 3),
        "avg_vertical_ratio": round(avg_vertical_ratio, 3),
    })

    
    # Visualization - pupils should move with your eye movement
    cv2.circle(frame, tuple(left_pupil.astype(int)), 4, (0, 255, 0), -1)
    cv2.circle(frame, tuple(right_pupil.astype(int)), 4, (0, 255, 0), -1)
    cv2.circle(frame, tuple(left_inner.astype(int)), 2, (255, 0, 0), -1)
    cv2.circle(frame, tuple(left_outer.astype(int)), 2, (255, 0, 0), -1)
    cv2.circle(frame, tuple(right_inner.astype(int)), 2, (255, 0, 0), -1)
    cv2.circle(frame, tuple(right_outer.astype(int)), 2, (255, 0, 0), -1)
    
    # Draw lines to show eye width
    cv2.line(frame, tuple(left_inner.astype(int)), tuple(left_outer.astype(int)), (0, 255, 255), 1)
    cv2.line(frame, tuple(right_inner.astype(int)), tuple(right_outer.astype(int)), (0, 255, 255), 1)
    
    cv2.putText(frame, f"Eye Ratio: {avg_ratio:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"L: {left_ratio:.2f} R: {right_ratio:.2f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    

    
    return direction

def print_eye_direction_report(eye_tracking_log):
    total_entries = len(eye_tracking_log)
    if total_entries == 0:
        print("No eye tracking data available.")
        return
    
    direction_counts = {"LEFT": 0, "RIGHT": 0, "CENTER": 0, "UP": 0, "DOWN": 0, "unknown": 0}

    
    for entry in eye_tracking_log:
        direction = entry.get("direction", "unknown")
        if direction in direction_counts:
            direction_counts[direction] += 1
        else:
            direction_counts["unknown"] += 1
    
    print("\n📊 Eye Direction Summary Report:")
    for direction, count in direction_counts.items():
        percentage = (count / total_entries) * 100
        print(f"  {direction:>6}: {count:>4} frames ({percentage:5.1f}%)")
    return {"total": total_entries, "direction_counts": direction_counts}

