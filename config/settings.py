# Threshold settings for exercise detection

# Stationary Running
RUNNING_KNEE_HEIGHT_THRESHOLD = 0.05  # Percentage threshold above resting position

# Push-ups
PUSHUP_ELBOW_ANGLE_THRESHOLD = 90.0  # Degrees
PUSHUP_BODY_HORIZONTAL_THRESHOLD = 0.1  # Acceptable difference between shoulder and hip height

# Squats
SQUAT_KNEE_ANGLE_THRESHOLD = 120.0  # Degrees

# Jumping Jacks
JUMPING_JACK_ARM_THRESHOLD = 100.0  # Degrees
JUMPING_JACK_LEG_THRESHOLD = 30.0  # Degrees

# Exercise modes
EXERCISE_MODES = {
    0: "Calibration",
    1: "Stationary Running",
    2: "Push-ups",
    3: "Squats",
    4: "Jumping Jacks"
}

# MediaPipe settings
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
