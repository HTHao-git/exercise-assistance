import mediapipe as mp
from exercise_detectors.base_detector import BaseExerciseDetector
from utils.angle_utils import calculate_angle
from config.settings import SQUAT_KNEE_ANGLE_THRESHOLD

mp_pose = mp.solutions.pose

class SquatDetector(BaseExerciseDetector):
    def __init__(self):
        super().__init__("Squats")
        self.knee_angle_threshold = SQUAT_KNEE_ANGLE_THRESHOLD
        
    def detect(self, landmarks):
        if not landmarks:
            return self.counter, "No landmarks"
            
        # Get hip, knee, and ankle points for angle calculation
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
              landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Calculate knee angle
        knee_angle = calculate_angle(hip, knee, ankle)
        
        # Determine squat stage based on knee angle
        if self.stage == None or self.stage == 'up':
            if knee_angle < self.knee_angle_threshold:
                self.stage = 'down'
                
        elif self.stage == 'down':
            if knee_angle > self.knee_angle_threshold:
                self.stage = 'up'
                self.counter += 1
                
        return self.counter, self.stage