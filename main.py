import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Import from project modules
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
    display_controls
)
from config.settings import EXERCISE_MODES, POSE_MIN_DETECTION_CONFIDENCE, POSE_MIN_TRACKING_CONFIDENCE
from pose_guide import PoseGuide  # Import the new PoseGuide class

class ExerciseRecognitionSystem:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE)
        
        # Initialize exercise detectors
        self.detectors = {
            1: StationaryRunningDetector(),
            2: PushupDetector(),
            3: SquatDetector(),
            4: JumpingJackDetector()
        }
        
        self.exercise_mode = 0  # Start with calibration mode
        
        # Add calibration state variables
        self.calibrating = False
        self.countdown_start = None
        self.countdown_seconds = 10  # Extended to 10 seconds for reading instructions
        self.instruction_seconds = 7  # Time to show instructions
        self.capture_start = None
        self.capture_seconds = 3
        self.calibration_message = ""
        
        # Initialize the pose guide
        self.pose_guide = PoseGuide()
        
    def process_frame(self, frame):
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose
        result = self.pose.process(rgb_frame)
        
        count = 0
        stage = "Unknown"
        
        # Draw skeleton on the frame if landmarks detected
        if result.pose_landmarks:
            # Draw landmarks
            frame = draw_landmarks(frame, result.pose_landmarks, self.mp_pose, self.mp_drawing)
            
            landmarks = result.pose_landmarks.landmark
            
            # Process the current exercise mode (if not calibrating)
            if not self.calibrating:
                if self.exercise_mode == 0:  # Calibration mode
                    status_message = "Select an exercise using 'm' first, then press 'c' to calibrate."
                    frame = draw_calibration_status(frame, status_message)
                else:
                    # Get the current detector
                    detector = self.detectors.get(self.exercise_mode)
                    if detector and detector.calibrated:
                        count, stage = detector.detect(landmarks)
                    elif detector:
                        status_message = "Press 'c' to calibrate for this exercise."
                        frame = draw_calibration_status(frame, status_message)
                
        # Draw exercise information
        exercise_name = EXERCISE_MODES.get(self.exercise_mode, "Unknown")
        frame = draw_exercise_info(frame, exercise_name, count, stage)
        
        # Process calibration overlay (countdown, etc.)
        frame = self.process_calibration(frame, result.pose_landmarks.landmark if result.pose_landmarks else None)
        
        return frame
        
    def change_exercise_mode(self):
        """Switch exercise mode and reset any ongoing calibration"""
        # End any ongoing calibration
        self.calibrating = False
        self.countdown_start = None
        self.capture_start = None
        
        # Cycle through exercise modes
        self.exercise_mode = (self.exercise_mode + 1) % len(EXERCISE_MODES)
        
        # Reset the detector for the new exercise mode
        if self.exercise_mode in self.detectors:
            self.detectors[self.exercise_mode] = type(self.detectors[self.exercise_mode])()
        
    def calibrate_current_detector(self):
        """Start the calibration countdown process"""
        if self.exercise_mode == 0:
            # In calibration mode, do nothing
            return False
        
        # Start calibration countdown
        self.calibrating = True
        self.countdown_start = time.time()
        self.calibration_message = "Read the instructions carefully..."
        return True
        
    def process_calibration(self, frame, landmarks=None):
        """Process calibration countdown and capture"""
        current_time = time.time()
        
        # In capture phase - check this first
        if self.capture_start is not None:
            elapsed = current_time - self.capture_start
            remaining = max(0, self.capture_seconds - int(elapsed))
            
            # Display capture progress
            self.calibration_message = f"Hold position! Capturing... {remaining}"
            
            # Try to calibrate with current frame
            if landmarks and elapsed >= 1.0:  # Capture one frame per second during capture
                # Extract key points for the specific exercise
                key_points = self.pose_guide.extract_key_points(landmarks, self.exercise_mode)
                
                # Try to calibrate with current frame
                detector = self.detectors.get(self.exercise_mode)
                if detector:
                    # Pass both landmarks and key points for more detailed calibration
                    detector.calibrate(landmarks, key_points=key_points)
            
            # Capture finished
            if elapsed >= self.capture_seconds:
                self.capture_start = None
                self.calibrating = False
                
                detector = self.detectors.get(self.exercise_mode)
                if detector and landmarks:  # Only mark as calibrated if landmarks were detected
                    detector.calibrated = True
                    self.calibration_message = "Calibration complete!"
                else:
                    # No landmarks detected during calibration
                    self.calibration_message = "Calibration failed! No body detected."
        
        # In instructions and countdown phase
        elif self.countdown_start is not None:
            elapsed = current_time - self.countdown_start
            remaining = max(0, self.countdown_seconds - int(elapsed))
            
            # Show pose instructions for the first part of the countdown
            if elapsed <= self.instruction_seconds:
                # Draw pose-specific instructions
                frame = self.pose_guide.draw_pose_instructions(frame, self.exercise_mode, remaining)
            else:
                # Display countdown only
                self.calibration_message = f"Get ready! {remaining}..."
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, self.calibration_message, 
                            (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Countdown finished, start capture
            if elapsed >= self.countdown_seconds:
                self.countdown_start = None
                self.capture_start = current_time
                self.calibration_message = "Hold position! Capturing..."
        
        # Draw calibration message only if not in instruction display phase
        if self.countdown_start is None or current_time - self.countdown_start > self.instruction_seconds:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, self.calibration_message, 
                        (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame
        
        # In instructions and countdown phase
        if self.countdown_start is not None:
            elapsed = current_time - self.countdown_start
            remaining = max(0, self.countdown_seconds - int(elapsed))
            
            # Show pose instructions for the first part of the countdown
            if elapsed <= self.instruction_seconds:
                # Draw pose-specific instructions
                frame = self.pose_guide.draw_pose_instructions(frame, self.exercise_mode, remaining)
            else:
                # Display countdown only
                self.calibration_message = f"Get ready! {remaining}..."
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, self.calibration_message, 
                            (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Countdown finished, start capture
            if elapsed >= self.countdown_seconds:
                self.countdown_start = None
                self.capture_start = current_time
                self.calibration_message = "Hold position! Capturing..."
        
        # In capture phase
        elif self.capture_start is not None:
            elapsed = current_time - self.capture_start
            remaining = max(0, self.capture_seconds - int(elapsed))
            
            # Display capture progress
            self.calibration_message = f"Hold position! Capturing... {remaining}"
            
            # Try to calibrate with current frame
            if landmarks and elapsed >= 1.0:  # Capture one frame per second during capture
                # Extract key points for the specific exercise
                key_points = self.pose_guide.extract_key_points(landmarks, self.exercise_mode)
                
                # Try to calibrate with current frame
                detector = self.detectors.get(self.exercise_mode)
                if detector:
                    # Pass both landmarks and key points for more detailed calibration
                    detector.calibrate(landmarks, key_points=key_points)
            
            # Capture finished
            if elapsed >= self.capture_seconds:
                self.capture_start = None
                self.calibrating = False
                self.calibration_message = "Calibration complete!"
                detector = self.detectors.get(self.exercise_mode)
                if detector and landmarks:  # Only mark as calibrated if landmarks were detected
                    detector.calibrated = True
                else:
                    # No landmarks detected during calibration
                    self.calibration_message = "Calibration failed! No body detected."
        
        # Draw calibration message only if not in instruction display phase
        if self.countdown_start is None or current_time - self.countdown_start > self.instruction_seconds:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, self.calibration_message, 
                        (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame
    
def main():
    panel_width = 400
    cam_width = 1200
    cam_height = 800

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Initialize exercise recognition system
    exercise_system = ExerciseRecognitionSystem()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (cam_width, cam_height))
        frame = cv2.flip(frame, 1)

        # Process the frame (draws pose, exercise info, etc.)
        processed_frame = exercise_system.process_frame(frame)

        # Create a canvas and paste the processed frame in the center
        canvas = np.zeros((cam_height, cam_width + 2 * panel_width, 3), dtype=np.uint8)
        canvas[:, panel_width:panel_width + cam_width] = processed_frame

        # Left panel: Instructions
        cv2.putText(canvas, "Instructions:", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Press 'm' to change exercise", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(canvas, "Press 'c' to calibrate", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(canvas, "Press 'q' to quit", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # Right panel: Dynamic exercise information
        cv2.putText(canvas, "Current Exercise:", (cam_width + panel_width + 10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(canvas, EXERCISE_MODES.get(exercise_system.exercise_mode, "None"), 
                   (cam_width + panel_width + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        
        # Show calibration status
        calibrated = False
        if exercise_system.exercise_mode in exercise_system.detectors:
            detector = exercise_system.detectors[exercise_system.exercise_mode]
            calibrated = detector.calibrated
            
        status = "Calibrated" if calibrated else "Not Calibrated"
        color = (0, 255, 0) if calibrated else (0, 0, 255)
        cv2.putText(canvas, f"Status: {status}", (cam_width + panel_width + 10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show the canvas
        cv2.imshow('Exercise Recognition', canvas)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            exercise_system.change_exercise_mode()
        elif key == ord('c'):
            exercise_system.calibrate_current_detector()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()