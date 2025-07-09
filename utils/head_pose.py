import cv2
import numpy as np
import time
from utils.config_loader import load_config
config = load_config()
head_pose_log = []
def simple_head_pose_detection(frame, face_mesh , head_pose_log=head_pose_log):
    image = cv2.cvtColor(cv2.flip(frame.copy(), 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, _ = image.shape
    face_3d = []
    face_2d = []

    landmark_ids = [33, 263, 1, 61, 291, 199]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx in landmark_ids:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                if idx == 1:
                    nose_2d = (x, y)
                    nose_3d = (x, y, lm.z * 3000)

                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

            if len(face_2d) < 6 or len(face_3d) < 6:
                return "neutral"

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = img_w
            cam_matrix = np.array([
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w / 2],
                [0, 0, 1]
            ])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            x_angle = angles[0] * 360
            y_angle = angles[1] * 360
            z_angle = angles[2] * 360

            # 👇 Adjust sensitivity here
            if y_angle < config["head_yaw_left_threshold"]:
                text = "Looking Left"
            elif y_angle > config["head_yaw_right_threshold"]:
                text = "Looking Right"
            elif x_angle < config["head_pitch_down_threshold"]:  # Less sensitive than >10
                text = "Looking Down"
            elif x_angle > config["head_pitch_up_threshold"]     :  # Less sensitive than >10
                text = "Looking Up"
            else:
                text = "Forward"

            # Nose direction vector
            nose_3d_projection, _ = cv2.projectPoints(
                np.array([nose_3d]), rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            head_pose_log.append({
                "timestamp": time.time(),
                "x_angle": round(x_angle, 1),
                "y_angle": round(y_angle, 1),
                "z_angle": round(z_angle, 1),
                "direction": text
            })
            # Draw pose info
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 2)
            cv2.putText(image, f"x: {round(x_angle, 1)}", (400, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"y: {round(y_angle, 1)}", (400, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"z: {round(z_angle, 1)}", (400, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Copy result back to original frame
            frame[:] = image[:]
            return text

    return "neutral"

def print_head_pose_report(head_pose_log):
    total_entries = len(head_pose_log)
    if total_entries == 0:
        print("No head tracking data available.")
        return
    
    direction_counts = {"Looking Left": 0, "Looking Right": 0, "Looking Down": 0, "Looking Up": 0, "Forward": 0}
    
    for entry in head_pose_log:
        direction = entry.get("direction", "unknown")
        if direction in direction_counts:
            direction_counts[direction] += 1
        else:
            direction_counts["unknown"] += 1
    
    print("\n📊 Head Direction Summary Report:")
    for direction, count in direction_counts.items():
        percentage = (count / total_entries) * 100
        print(f"  {direction:>6}: {count:>4} frames ({percentage:5.1f}%)")
    return {"total": total_entries, "direction_counts": direction_counts}

    