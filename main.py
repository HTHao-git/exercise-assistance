import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk

# Project module imports
from exercise_detectors import (
    StationaryRunningDetector,
    PushupDetector,
    SquatDetector,
    JumpingJackDetector
)
from utils.visualization import (
    draw_landmarks,
    draw_exercise_info,
    draw_calibration_status,
)
from config.settings import EXERCISE_MODES, POSE_MIN_DETECTION_CONFIDENCE, POSE_MIN_TRACKING_CONFIDENCE
from pose_guide import PoseGuide

def get_screen_size():
    try:
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return 1280, 720

PANEL_WIDTH = 400

class ExerciseRecognitionSystem:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE)

        self.detectors = {
            1: StationaryRunningDetector(),
            2: PushupDetector(),
            3: SquatDetector(),
            4: JumpingJackDetector()
        }

        self.exercise_mode = 0  # Start with camera test mode

        self.calibrating = False
        self.countdown_start = None
        self.countdown_seconds = 10
        self.instruction_seconds = 7
        self.capture_start = None
        self.capture_seconds = 3
        self.calibration_message = ""

        self.pose_guide = PoseGuide()
        self.latest_landmarks = None
        self.debug_mode = False

    def process_frame(self, frame, cam_width, cam_height):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)
        count = 0
        stage = "Unknown"

        if result.pose_landmarks:
            frame = draw_landmarks(frame, result.pose_landmarks, self.mp_pose, self.mp_drawing)
            self.latest_landmarks = result.pose_landmarks.landmark
            if not self.calibrating:
                if self.exercise_mode == 0:
                    status_message = "Select an exercise using 'm' or press 'c' to start camera testing."
                    frame = draw_calibration_status(frame, status_message)
                else:
                    detector = self.detectors.get(self.exercise_mode)
                    if detector and getattr(detector, 'calibrated', False):
                        count, stage = detector.detect(self.latest_landmarks)
                    elif detector:
                        status_message = "Press 'c' to calibrate for this exercise."
                        frame = draw_calibration_status(frame, status_message)
        else:
            self.latest_landmarks = None

        frame = draw_exercise_info(frame, EXERCISE_MODES.get(self.exercise_mode, "Unknown"), count, stage)
        frame = self.process_calibration(frame)

        # Draw debug line for stationary running's ANKLE threshold
        if self.debug_mode and self.exercise_mode == 1:
            detector = self.detectors.get(1)
            if detector and hasattr(detector, 'min_ankle_height') and hasattr(detector, 'ankle_height_threshold') and detector.min_ankle_height is not None:
                threshold_y = detector.min_ankle_height - detector.ankle_height_threshold
                threshold_y = min(max(threshold_y, 0.0), 1.0)
                line_y = int(threshold_y * cam_height)
                cv2.line(frame, (0, line_y), (cam_width, line_y), (0, 255, 255), 2)
                cv2.putText(frame, "Ankle Raise Threshold", (10, line_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame

    def change_exercise_mode(self):
        self.calibrating = False
        self.countdown_start = None
        self.capture_start = None
        self.exercise_mode = (self.exercise_mode + 1) % len(EXERCISE_MODES)
        if self.exercise_mode in self.detectors:
            self.detectors[self.exercise_mode] = type(self.detectors[self.exercise_mode])()
            # Reset latest_landmarks to avoid carrying over

    def calibrate_current_detector(self):
        if self.exercise_mode == 0:
            self.calibrating = True
            self.calibration_message = "Camera testing active. Press 'c' again to finish."
            self.countdown_start = None
            self.capture_start = None
        else:
            self.countdown_seconds = 10
            self.instruction_seconds = 7
            self.calibrating = True
            self.countdown_start = time.time()
            self.capture_start = None
            self.calibration_message = "Read the instructions carefully..."
        return True

    def process_calibration(self, frame):
        current_time = time.time()
        if self.exercise_mode == 0 and self.calibrating:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, self.calibration_message,
                        (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)
            return frame

        # Calibration phase for exercise
        if self.capture_start is not None:
            elapsed = current_time - self.capture_start
            remaining = max(0, self.capture_seconds - int(elapsed))

            self.calibration_message = f"Hold position! Testing... {remaining}"

            # Capture finished
            detector = self.detectors.get(self.exercise_mode)
            if detector and self.latest_landmarks:
                calibrated_ok = detector.calibrate(self.latest_landmarks)
                if calibrated_ok:
                    self.calibration_message = "Calibration complete!"
                else:
                    self.calibration_message = "Calibration failed! No body detected."
            else:
                self.calibration_message = "Calibration failed! No body detected."
            self.calibrating = False
            self.capture_start = None

        elif self.countdown_start is not None:
            elapsed = current_time - self.countdown_start
            remaining = max(0, self.countdown_seconds - int(elapsed))
            if elapsed <= self.instruction_seconds:
                frame = self.pose_guide.draw_pose_instructions(frame, self.exercise_mode, remaining)
            else:
                self.calibration_message = f"Get ready! {remaining}..."
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, self.calibration_message,
                            (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2, cv2.LINE_AA)
            if elapsed >= self.countdown_seconds:
                self.countdown_start = None
                self.capture_start = current_time
                self.calibration_message = "Hold position! Capturing..."

        if (self.countdown_start is None or
            (self.countdown_start and current_time - self.countdown_start > self.instruction_seconds)):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, self.calibration_message,
                        (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

def main():
    screen_width, screen_height = get_screen_size()
    panel_width = PANEL_WIDTH
    cam_width = max(400, screen_width - 2 * panel_width)
    cam_height = int(screen_height * 0.8)

    cap = cv2.VideoCapture(0)
    exercise_system = ExerciseRecognitionSystem()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (cam_width, cam_height))
        frame = cv2.flip(frame, 1)
        processed_frame = exercise_system.process_frame(frame, cam_width, cam_height)
        if processed_frame is None:
            processed_frame = np.zeros_like(frame)  # fallback to blank

        canvas = np.zeros((cam_height, cam_width + 2 * panel_width, 3), dtype=np.uint8)
        canvas[:, panel_width:panel_width + cam_width] = processed_frame

        # Left panel: Instructions
        cv2.putText(canvas, "Instructions:", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Press 'm' to change exercise", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(canvas, "Press 'c' to calibrate", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(canvas, "Press 'd' to toggle debug info", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(canvas, "Press 'q' to quit", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # Right panel: Status and debug
        cv2.putText(canvas, "Current Exercise:", (cam_width + panel_width + 10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(canvas, EXERCISE_MODES.get(exercise_system.exercise_mode, "None"),
                   (cam_width + panel_width + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        calibrated = False
        if exercise_system.exercise_mode in exercise_system.detectors:
            detector = exercise_system.detectors[exercise_system.exercise_mode]
            calibrated = getattr(detector, 'calibrated', False)
        status = "Calibrated" if calibrated else "Not Calibrated"
        color = (0, 255, 0) if calibrated else (0, 0, 255)
        cv2.putText(canvas, f"Status: {status}", (cam_width + panel_width + 10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Body parts, debug info, etc. (as in your previous main)
        if exercise_system.debug_mode:
            if exercise_system.exercise_mode in exercise_system.detectors:
                detector = exercise_system.detectors[exercise_system.exercise_mode]
                debug_info = getattr(detector, "last_debug_info", None)
                if debug_info:
                    y_offset = 600
                    cv2.putText(canvas, "Detector Debug Info:", (cam_width + panel_width + 10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    for i, (key, val) in enumerate(debug_info.items()):
                        cv2.putText(canvas, f"{key}: {val}", (cam_width + panel_width + 10, y_offset + 30 + (i+1)*25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1)

        cv2.imshow('Exercise Recognition', canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            exercise_system.change_exercise_mode()
        elif key == ord('c'):
            if exercise_system.exercise_mode == 0 and exercise_system.calibrating:
                exercise_system.calibrating = False
                exercise_system.calibration_message = "Camera and position test ended."
            else:
                exercise_system.calibrate_current_detector()
        elif key == ord('d'):
            exercise_system.debug_mode = not exercise_system.debug_mode

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()