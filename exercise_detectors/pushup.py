import mediapipe as mp
from exercise_detectors.base_detector import BaseExerciseDetector
from utils.angle_utils import calculate_angle
from config.settings import PUSHUP_ELBOW_ANGLE_THRESHOLD, PUSHUP_BODY_HORIZONTAL_THRESHOLD

mp_pose = mp.solutions.pose

class PushupDetector(BaseExerciseDetector):
    def __init__(self):
        super().__init__("Push-ups")
        self.elbow_angle_threshold = PUSHUP_ELBOW_ANGLE_THRESHOLD
        self.body_horizontal_threshold = PUSHUP_BODY_HORIZONTAL_THRESHOLD
        
    def calibrate(self, landmarks, key_points=None):
        """
        Calibrate the push-up detector with the user's specific body proportions
        
        Args:
            landmarks: MediaPipe pose landmarks
            key_points: Optional dictionary with extracted key points for push-ups
            
        Returns:
            bool: True if calibration successful, False otherwise
        """
        # First call the parent class implementation to store key_points
        super().calibrate(landmarks, key_points)
        
        if landmarks and key_points:
            # If specific key points for push-ups were provided, use them
            if 'body_alignment' in key_points:
                # Store the reference body alignment for horizontal position
                self.reference_body_alignment = key_points['body_alignment']
                # Could adjust the threshold based on the user's specific body proportions
                self.body_horizontal_threshold = max(0.05, self.reference_body_alignment * 1.5)
                
            if 'left_elbow' in key_points and 'right_elbow' in key_points:
                # Could customize elbow angle threshold based on user's arm proportions
                # (this is just an example, might need adjustment)
                pass
                
            return True
            
        return landmarks is not None
        
    def detect(self, landmarks):
        if not landmarks:
            return self.counter, "No landmarks"
        
        # Get relevant landmarks
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # Calculate the elbow angle
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Check if the person is in the push-up position (we can add more checks here)
        shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        
        # Simple check if person is horizontal (shoulders and hips roughly at same height)
        is_horizontal = abs(shoulder_y - hip_y) < self.body_horizontal_threshold
        
        if not is_horizontal:
            return self.counter, "Not in position"
        
        # Check push-up stages based on elbow angle
        if self.stage == None or self.stage == 'up':
            if elbow_angle < self.elbow_angle_threshold:
                self.stage = 'down'
                
        elif self.stage == 'down':
            if elbow_angle > self.elbow_angle_threshold:
                self.stage = 'up'
                self.counter += 1
                
        return self.counter, self.stage