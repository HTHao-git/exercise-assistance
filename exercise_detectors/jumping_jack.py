import mediapipe as mp
from exercise_detectors.base_detector import BaseExerciseDetector
from utils.angle_utils import calculate_angle
from config.settings import JUMPING_JACK_ARM_THRESHOLD, JUMPING_JACK_LEG_THRESHOLD

mp_pose = mp.solutions.pose

class JumpingJackDetector(BaseExerciseDetector):
    def __init__(self):
        super().__init__("Jumping Jacks")
        self.arm_threshold = JUMPING_JACK_ARM_THRESHOLD
        self.leg_threshold = JUMPING_JACK_LEG_THRESHOLD
        self.last_debug_info = {}
        self.base_hip_width = None

    def calibrate(self, landmarks, key_points=None):
        super().calibrate(landmarks, key_points)
        if landmarks:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            # Use hip width as normalization baseline
            self.base_hip_width = abs(left_hip.x - right_hip.x)
            return True
        return False

    def detect(self, landmarks):
        if not landmarks:
            self.last_debug_info = {"status": "No landmarks"}
            return self.counter, "No landmarks"
        try:
            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        except (IndexError, AttributeError):
            self.last_debug_info = {"status": "Keypoints missing"}
            return self.counter, "Keypoints missing"

        # Arm angles
        arm_angle_left = calculate_angle(right_shoulder, left_shoulder, left_wrist)
        arm_angle_right = calculate_angle(left_shoulder, right_shoulder, right_wrist)
        arm_angle = (arm_angle_left + arm_angle_right) / 2

        # Leg spread: use normalized ankle distance
        ankle_dist = abs(left_ankle[0] - right_ankle[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        # Use calibration baseline if available, else use current hip width
        norm_ankle_dist = ankle_dist / (self.base_hip_width if self.base_hip_width else hip_width)

        # DEBUG INFO for right panel
        self.last_debug_info = {
            "arm_angle_left": round(arm_angle_left, 2),
            "arm_angle_right": round(arm_angle_right, 2),
            "arm_angle_avg": round(arm_angle, 2),
            "arm_threshold": round(self.arm_threshold, 2),
            "ankle_dist": round(ankle_dist, 4),
            "hip_width": round(hip_width, 4),
            "norm_ankle_dist": round(norm_ankle_dist, 2),
            "leg_threshold": round(self.leg_threshold, 2),
            "stage": self.stage,
            "counter": self.counter
        }

        # Jumping jack stage logic
        # "Up": arms up (angle > threshold) and legs apart (ankle_dist > threshold)
        # "Down": arms down (angle < threshold) and legs together (ankle_dist < threshold)
        if self.stage is None or self.stage == 'down':
            if arm_angle > self.arm_threshold and norm_ankle_dist > self.leg_threshold:
                self.stage = 'up'
        elif self.stage == 'up':
            if arm_angle < self.arm_threshold and norm_ankle_dist < self.leg_threshold:
                self.stage = 'down'
                self.counter += 1

        return self.counter, self.stage