import mediapipe as mp
from exercise_detectors.base_detector import BaseExerciseDetector
from utils.angle_utils import calculate_angle
from config.settings import SQUAT_KNEE_ANGLE_THRESHOLD

mp_pose = mp.solutions.pose

class SquatDetector(BaseExerciseDetector):
    def __init__(self):
        super().__init__("Squats")
        self.knee_angle_threshold = SQUAT_KNEE_ANGLE_THRESHOLD
        self.last_debug_info = {}

    def calibrate(self, landmarks, key_points=None):
        super().calibrate(landmarks, key_points)
        if landmarks and key_points:
            if 'hip_height' in key_points:
                self.reference_hip_height = key_points['hip_height']
            if 'hip_ankle_distance' in key_points:
                self.reference_hip_ankle_distance = key_points['hip_ankle_distance']
            return True
        return landmarks is not None

    def detect(self, landmarks):
        if not landmarks:
            self.last_debug_info = {"status": "No landmarks"}
            return self.counter, "No landmarks"
        
        # Get hip, knee, and ankle points for angle calculation (Left side)
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate knee angle
        knee_angle = calculate_angle(hip, knee, ankle)

        # DEBUG INFO for live panel
        self.last_debug_info = {
            "knee_angle": round(knee_angle, 2),
            "knee_angle_threshold": round(self.knee_angle_threshold, 2),
            "hip_y": round(hip[1], 4),
            "knee_y": round(knee[1], 4),
            "ankle_y": round(ankle[1], 4),
            "stage": self.stage,
            "counter": self.counter
        }

        # Determine squat stage based on knee angle
        if self.stage is None or self.stage == 'up':
            if knee_angle < self.knee_angle_threshold:
                self.stage = 'down'
        elif self.stage == 'down':
            if knee_angle > self.knee_angle_threshold:
                self.stage = 'up'
                self.counter += 1

        return self.counter, self.stage