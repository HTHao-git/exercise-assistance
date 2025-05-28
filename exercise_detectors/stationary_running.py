import mediapipe as mp
from exercise_detectors.base_detector import BaseExerciseDetector
from config.settings import RUNNING_ANKLE_HEIGHT_THRESHOLD

mp_pose = mp.solutions.pose

class StationaryRunningDetector(BaseExerciseDetector):
    def __init__(self):
        super().__init__("Stationary Running")
        self.min_ankle_height = None  # Calibrated once
        self.ankle_height_threshold = RUNNING_ANKLE_HEIGHT_THRESHOLD  # e.g., 0.10
        self.last_debug_info = {}

    def calibrate(self, landmarks, key_points=None):
        """Calibrate the ankle height threshold based on the user's standing position."""
        super().calibrate(landmarks, key_points)
        if landmarks:
            left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            self.min_ankle_height = (left_ankle_y + right_ankle_y) / 2
            self.calibrated = True
            print(f"[CALIBRATION] min_ankle_height: {self.min_ankle_height:.4f}")
            return True
        self.calibrated = False
        return False

    def detect(self, landmarks):
        if not landmarks or not self.calibrated or self.min_ankle_height is None:
            self.last_debug_info = {"status": "Not calibrated"}
            return self.counter, "Not calibrated"

        left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        threshold = self.min_ankle_height - self.ankle_height_threshold

        self.last_debug_info = {
            "left_ankle_y": round(left_ankle_y, 4),
            "right_ankle_y": round(right_ankle_y, 4),
            "min_ankle_height": round(self.min_ankle_height, 4),
            "threshold": round(threshold, 4),
            "stage": self.stage,
            "counter": self.counter
        }

        # State machine
        if self.stage is None or self.stage == 'down':
            if left_ankle_y < threshold or right_ankle_y < threshold:
                self.stage = 'up'
        elif self.stage == 'up':
            if left_ankle_y > threshold and right_ankle_y > threshold:
                self.stage = 'down'
                self.counter += 1

        return self.counter, self.stage