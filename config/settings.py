# Threshold settings for exercise detection

# Stationary Running
RUNNING_ANKLE_HEIGHT_THRESHOLD = 0.07  # Acceptable difference in ankle height for a step

# Push-ups
PUSHUP_ELBOW_ANGLE_THRESHOLD = 90.0  # Degrees
PUSHUP_BODY_HORIZONTAL_THRESHOLD = 0.1  # Acceptable difference between shoulder and hip height

# Squats
SQUAT_KNEE_ANGLE_THRESHOLD = 120.0  # Degrees

# Jumping Jacks
JUMPING_JACK_ARM_THRESHOLD = 86.0  # Degrees
JUMPING_JACK_LEG_THRESHOLD = 1.15

# Exercise modes
EXERCISE_MODES = {
    0: "Camera and Position Testing",
    1: "Stationary Running",
    2: "Push-up",
    3: "Squat",
    4: "Jumping Jack",
}

# MediaPipe settings
POSE_MIN_DETECTION_CONFIDENCE = 0.4
POSE_MIN_TRACKING_CONFIDENCE = 0.4
