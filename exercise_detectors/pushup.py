import mediapipe as mp
from exercise_detectors.base_detector import BaseExerciseDetector
from config.settings import PUSHUP_ELBOW_ANGLE_THRESHOLD, PUSHUP_BODY_HORIZONTAL_THRESHOLD

mp_pose = mp.solutions.pose

class PushupDetector(BaseExerciseDetector):
    def __init__(self):
        super().__init__("Push-ups")
        self.elbow_angle_threshold = PUSHUP_ELBOW_ANGLE_THRESHOLD
        self.body_horizontal_threshold = PUSHUP_BODY_HORIZONTAL_THRESHOLD
        self.last_debug_info = {}

    def calibrate(self, landmarks, key_points=None):
        super().calibrate(landmarks, key_points)
        if landmarks and key_points:
            if 'body_alignment' in key_points:
                self.reference_body_alignment = key_points['body_alignment']
                self.body_horizontal_threshold = max(0.05, self.reference_body_alignment * 1.5)
            # Could add more calibration with elbows etc.
            return True
        return landmarks is not None

    def detect(self, landmarks):
        if not landmarks:
            self.last_debug_info = {"status": "No landmarks"}
            return self.counter, "No landmarks"

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

        shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        is_horizontal = abs(shoulder_y - hip_y) < self.body_horizontal_threshold

        # DEBUG INFO
        self.last_debug_info = {
            "elbow_angle": round(elbow_angle, 2),
            "elbow_angle_threshold": round(self.elbow_angle_threshold, 2),
            "shoulder_y": round(shoulder_y, 4),
            "hip_y": round(hip_y, 4),
            "body_horizontal_threshold": round(self.body_horizontal_threshold, 4),
            "is_horizontal": is_horizontal,
            "stage": self.stage,
            "counter": self.counter
        }

        if not is_horizontal:
            return self.counter, "Not in position"

        if self.stage is None or self.stage == 'up':
            if elbow_angle < self.elbow_angle_threshold:
                self.stage = 'down'
        elif self.stage == 'down':
            if elbow_angle > self.elbow_angle_threshold:
                self.stage = 'up'
                self.counter += 1

        return self.counter, self.stage