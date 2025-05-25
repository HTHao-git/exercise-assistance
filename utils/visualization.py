import cv2

def draw_landmarks(frame, pose_landmarks, mp_pose, mp_drawing):
    """Draw skeleton landmarks on the frame"""
    mp_drawing.draw_landmarks(
        frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )
    return frame

def draw_exercise_info(frame, exercise_name, count, stage):
    """Draw exercise information on the frame"""
    cv2.putText(frame, f"Exercise: {exercise_name}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Count: {count}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Stage: {stage}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

def draw_calibration_status(frame, status_message):
    """Draw calibration status on the frame"""
    cv2.putText(frame, status_message, 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def display_controls(frame):
    """Display control instructions on the frame"""
    cv2.putText(frame, "Press 'm' to change exercise mode", 
                (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press 'c' to calibrate", 
                (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to quit", 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame