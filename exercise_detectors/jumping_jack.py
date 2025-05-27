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
        
    def calibrate(self, landmarks, key_points=None):
        """
        Calibrate the jumping jack detector with the user's specific body proportions
        
        Args:
            landmarks: MediaPipe pose landmarks
            key_points: Optional dictionary with extracted key points for jumping jacks
            
        Returns:
            bool: True if calibration successful, False otherwise
        """
        # First call the parent class implementation to store key_points
        super().calibrate(landmarks, key_points)
        
        if landmarks and key_points:
            # If specific key points for jumping jacks were provided, use them
            if 'shoulder_width' in key_points:
                # Use the resting shoulder width as reference
                self.base_shoulder_width = key_points['shoulder_width']
                
            if 'hip_width' in key_points:
                # Use the resting hip width as reference
                self.base_hip_width = key_points['hip_width']
                
            return True
            
        return landmarks is not None
    
    def detect(self, landmarks):
        if not landmarks:
            return self.counter, "No landmarks"
            
        # Get shoulder landmarks for arm position
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # Get hip landmarks for leg position
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Calculate angles
        # For arms: we measure the angle between the shoulders and wrists
        arm_angle_left = calculate_angle(right_shoulder, left_shoulder, left_wrist)
        arm_angle_right = calculate_angle(left_shoulder, right_shoulder, right_wrist)
        
        # For legs: we measure the angle between hips and ankles
        leg_angle = calculate_angle(left_hip, right_hip, right_ankle)
        
        # Average arm angle
        arm_angle = (arm_angle_left + arm_angle_right) / 2
        
        # Determine jumping jack stage
        # Arms up, legs apart = up position
        # Arms down, legs together = down position
        if self.stage == None or self.stage == 'down':
            if arm_angle > self.arm_threshold and leg_angle > self.leg_threshold:
                self.stage = 'up'
                
        elif self.stage == 'up':
            if arm_angle < self.arm_threshold and leg_angle < self.leg_threshold:
                self.stage = 'down'
                self.counter += 1
                
        return self.counter, self.stage