import cv2
import numpy as np
import mediapipe as mp

class PoseGuide:
    """
    Class to provide guidance for exercise calibration poses and extract relevant body points.
    """
    
    def __init__(self):
        # Dictionary of pose instructions for each exercise type
        self.pose_instructions = {
            0: {  # Camera and Position Testing0: {  # Camera and Position Testing
                "title": "Camera and Position Testing",
                "instructions": [
                    "Stand in front of the camera.",
                    "Ensure your whole body fits in the frame.",
                    "Check lighting: avoid strong shadows or glare.",
                    "You can move around to check camera coverage."
                ]
            },

            1: {  # Stationary Running
                "title": "Stationary Running Calibration",
                "instructions": [
                    "Stand straight with your feet shoulder-width apart",
                    "Let your arms rest naturally at your sides",
                    "Look straight ahead at the camera",
                    "This position will be used as your reference"
                ]
            },
            2: {  # Pushup
                "title": "Push-up Calibration",
                "instructions": [
                    "Get into the top push-up position (plank)",
                    "Arms straight, hands shoulder-width apart",
                    "Keep your body in a straight line",
                    "Face down with your head in a neutral position"
                ]
            },
            3: {  # Squat
                "title": "Squat Calibration",
                "instructions": [
                    "Stand straight with feet shoulder-width apart",
                    "Face the camera directly",
                    "Let your arms rest naturally at your sides",
                    "This position will be used as your standing reference"
                ]
            },
            4: {  # Jumping Jack
                "title": "Jumping Jack Calibration",
                "instructions": [
                    "Stand straight with feet together",
                    "Arms at your sides",
                    "Face the camera directly",
                    "This position will be used as your starting reference"
                ]
            }
        }
        
        # MediaPipe pose setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
    def get_pose_instructions(self, exercise_mode):
        """
        Get the calibration pose instructions for a specific exercise mode.
        
        Args:
            exercise_mode: The exercise mode number
            
        Returns:
            dict: Dictionary with title and instructions list, or None if not found
        """
        return self.pose_instructions.get(exercise_mode, None)
    
    def extract_key_points(self, landmarks, exercise_mode):
        """
        Extract key body points relevant to the specific exercise mode.
        
        Args:
            landmarks: MediaPipe pose landmarks
            exercise_mode: The exercise mode number
            
        Returns:
            dict: Dictionary containing key points and measurements
        """
        key_points = {}

        if landmarks is None:
            return key_points
        
        # Common measurements for all exercises
        # Store all key points as ratios to image dimensions for portability
        
        # Extract nose position (useful for most exercises)
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        key_points['nose'] = (nose.x, nose.y, nose.z)
        
        # Extract shoulders
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        key_points['left_shoulder'] = (left_shoulder.x, left_shoulder.y, left_shoulder.z)
        key_points['right_shoulder'] = (right_shoulder.x, right_shoulder.y, right_shoulder.z)
        
        # Extract hips
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        key_points['left_hip'] = (left_hip.x, left_hip.y, left_hip.z)
        key_points['right_hip'] = (right_hip.x, right_hip.y, right_hip.z)
        
        # Extract knees
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        key_points['left_knee'] = (left_knee.x, left_knee.y, left_knee.z)
        key_points['right_knee'] = (right_knee.x, right_knee.y, right_knee.z)
        
        # Extract ankles
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        key_points['left_ankle'] = (left_ankle.x, left_ankle.y, left_ankle.z)
        key_points['right_ankle'] = (right_ankle.x, right_ankle.y, right_ankle.z)
        
        # Extract wrists
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        key_points['left_wrist'] = (left_wrist.x, left_wrist.y, left_wrist.z)
        key_points['right_wrist'] = (right_wrist.x, right_wrist.y, right_wrist.z)
        
        # Exercise-specific measurements
        if exercise_mode == 1:  # Stationary Running - focus on knees and hips
            # Calculate standing knee height (average of left and right)
            knee_y_avg = (left_knee.y + right_knee.y) / 2
            hip_y_avg = (left_hip.y + right_hip.y) / 2
            key_points['knee_hip_distance'] = hip_y_avg - knee_y_avg
            key_points['standing_height'] = 1.0 - ((left_ankle.y + right_ankle.y) / 2)
            
        elif exercise_mode == 2:  # Pushup - focus on shoulders, elbows, wrists
            # Calculate arm extension and body alignment
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            key_points['left_elbow'] = (left_elbow.x, left_elbow.y, left_elbow.z)
            key_points['right_elbow'] = (right_elbow.x, right_elbow.y, right_elbow.z)
            
            # Calculate body alignment (straight line from shoulders to ankles)
            shoulder_y_avg = (left_shoulder.y + right_shoulder.y) / 2
            hip_y_avg = (left_hip.y + right_hip.y) / 2
            key_points['body_alignment'] = abs(shoulder_y_avg - hip_y_avg)
            
        elif exercise_mode == 3:  # Squat - focus on hip and knee angles
            # Calculate standing height and hip position
            hip_y_avg = (left_hip.y + right_hip.y) / 2
            ankle_y_avg = (left_ankle.y + right_ankle.y) / 2
            key_points['hip_height'] = 1.0 - hip_y_avg
            key_points['hip_ankle_distance'] = hip_y_avg - ankle_y_avg
            
        elif exercise_mode == 4:  # Jumping Jack - focus on shoulder and hip width
            # Calculate shoulder width and hip width
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)
            hip_width = abs(right_hip.x - left_hip.x)
            key_points['shoulder_width'] = shoulder_width
            key_points['hip_width'] = hip_width
            
        return key_points
        
    def draw_pose_instructions(self, frame, exercise_mode, time_remaining=10):
        """
        Draw pose instructions on the frame with a countdown timer.
        
        Args:
            frame: The video frame to draw on
            exercise_mode: The exercise mode number
            time_remaining: Remaining time in seconds for the countdown
            
        Returns:
            frame: The frame with instructions drawn
        """
        instructions = self.get_pose_instructions(exercise_mode)
        if not instructions:
            return frame
            
        # Create a semi-transparent overlay for better text readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw title
        cv2.putText(frame, "-" + instructions["title"], 
                   (int(frame.shape[1] * 0.1), int(frame.shape[0] * 0.15)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Draw instructions
        for i, instruction in enumerate(instructions["instructions"]):
            y_pos = int(frame.shape[0] * (0.25 + i * 0.08))
            cv2.putText(frame, instruction, 
                       (int(frame.shape[1] * 0.1), y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Draw countdown timer
        cv2.putText(frame, f"Time remaining: {time_remaining}s", 
                   (int(frame.shape[1] * 0.1), int(frame.shape[0] * 0.8)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Draw "Get Ready!" message
        if time_remaining <= 3:
            cv2.putText(frame, "Get Ready!", 
                       (int(frame.shape[1] * 0.3), int(frame.shape[0] * 0.9)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
        
        return frame