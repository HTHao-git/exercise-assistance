import mediapipe as mp
from exercise_detectors.base_detector import BaseExerciseDetector
from config.settings import RUNNING_KNEE_HEIGHT_THRESHOLD

mp_pose = mp.solutions.pose

class StationaryRunningDetector(BaseExerciseDetector):
    def __init__(self):
        super().__init__("Stationary Running")
        self.min_knee_height = 0  # Will be calibrated
        self.knee_height_threshold = RUNNING_KNEE_HEIGHT_THRESHOLD
        
    def calibrate(self, landmarks, key_points=None):
        """Calibrate the knee height threshold based on the user's standing position"""
        # First call the parent class implementation to store key_points
        super().calibrate(landmarks, key_points)

        if landmarks:
            # Get knee landmarks
            left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
            right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

            # Set the minimum knee height to current position (average of both knees)
            self.min_knee_height = (left_knee_y + right_knee_y) / 2
            self.calibrated = True
            return True
        return False
        
    def detect(self, landmarks):
        if not landmarks or not self.calibrated:
            return self.counter, "Not calibrated"
        
        # Get knee landmark y-coordinates (vertical position)
        left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        
        # Check if either knee is raised above threshold
        if self.stage == None or self.stage == 'down':
            if left_knee_y < self.min_knee_height - self.knee_height_threshold or right_knee_y < self.min_knee_height - self.knee_height_threshold:
                self.stage = 'up'
                
        elif self.stage == 'up':
            if left_knee_y > self.min_knee_height - (self.knee_height_threshold / 2) and right_knee_y > self.min_knee_height - (self.knee_height_threshold / 2):
                self.stage = 'down'
                self.counter += 1
                
        return self.counter, self.stage