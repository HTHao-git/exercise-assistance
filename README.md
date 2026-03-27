# exercise-assistance

A real-time **webcam-based exercise recognition and rep-counting** system built with **Python + OpenCV + MediaPipe Pose**.  
This project detects your body pose from a live camera feed, lets you **select an exercise mode**, **calibrate** to your body/starting pose, and then **counts repetitions** while displaying on-screen overlays (exercise name, rep count, stage, calibration prompts, and optional debug info).

## Features

- **Live pose tracking** using MediaPipe Pose
- **Real-time rep counting** with simple state machines per exercise
- **Multiple exercise modes**:
  - Camera and Position Testing
  - Stationary Running
  - Push-up
  - Squat
  - Jumping Jack
- **Per-exercise calibration** ('c' key) to establish a baseline/reference pose
- **On-screen instructions panels** + status display
- **Debug mode** ('d' key) to show internal detector measurements (thresholds, angles, etc.)

## Demo / UI Overview

When running, the app creates a wide canvas with:
- **Center**: live webcam feed + pose landmarks + exercise info overlay
- **Left panel**: keyboard controls / instructions
- **Right panel**: current exercise + calibration status + (optional) detector debug info

Key overlay info includes:
- 'Exercise: <name>'
- 'Count: <reps>'
- 'Stage: <up/down/...>'

## How It Works (High-level)

1. The webcam frame is captured with OpenCV ('cv2.VideoCapture(0)').
2. The frame is passed through **MediaPipe Pose** to extract human pose landmarks.
3. Depending on the selected mode:
   - If in test mode, it just shows pose + guidance.
   - If in an exercise mode:
     - You **calibrate** the detector (baseline pose/measurements).
     - The detector runs a **rep-counting state machine** based on angles/distances computed from landmarks.
4. The UI draws pose landmarks, rep count, stage, and calibration text.

## Exercise Modes

Exercise modes are defined in 'config/settings.py':

- '0': Camera and Position Testing
- '1': Stationary Running
- '2': Push-up
- '3': Squat
- '4': Jumping Jack

### Calibration guidance (PoseGuide)

'pose_guide.py' provides on-screen calibration instructions for each mode (e.g., "Stand straight…", "Top push-up position (plank)…").  
During calibration, the UI shows a countdown and then captures a short window to build baseline measurements.

## Detectors (Rep Counting Logic)

All detectors implement a common interface via `BaseExerciseDetector` (`exercise_detectors/base_detector.py`):

- 'detect(landmarks) -> (counter, stage)'
- 'calibrate(landmarks, key_points=None) -> bool'
- 'reset()'

Each detector maintains:
- 'counter': repetition count
- 'stage': current stage (usually "up" / "down")
- 'calibrated': whether calibration is complete
- 'calibration_data': dict of baseline measurements (optional)

### 1) Stationary Running ('exercise_detectors/stationary_running.py')
- **Calibration**: stores a baseline ankle height ('min_ankle_height') from standing still.
- **Detection**: counts a rep when ankles move **above a threshold** and return back down.
- Tracks debug info such as ankle y-values and the computed threshold.

Key setting:
- 'RUNNING_ANKLE_HEIGHT_THRESHOLD' (default '0.07')

### 2) Push-ups (`exercise_detectors/pushup.py`)
- **Detection signals**:
  - Elbow angle (computed from shoulder–elbow–wrist)
  - Body "horizontalness" check: 'abs(shoulder_y - hip_y) < threshold'
- **State machine**:
  - "up" → "down" when elbow angle drops below threshold
  - "down" → "up" when elbow angle rises above threshold, increments rep count
- Returns "Not in position" if the body horizontal check fails.

Key settings:
- 'PUSHUP_ELBOW_ANGLE_THRESHOLD' (default '90.0')
- 'PUSHUP_BODY_HORIZONTAL_THRESHOLD' (default '0.1')

> Note: 'PushupDetector' uses 'self.calculate_angle(...)' in the current code. Other detectors use 'utils/angle_utils.calculate_angle'. If you hit an AttributeError at runtime, that’s likely why—see "Troubleshooting".

### 3) Squats ('exercise_detectors/squat.py')
- Computes the **knee angle** from hip–knee–ankle (left side).
- **State machine**:
  - "up" → "down" when knee angle goes below threshold
  - "down" → "up" when knee angle rises above threshold, increments rep count

Key setting:
- 'SQUAT_KNEE_ANGLE_THRESHOLD' (default '120.0')

### 4) Jumping Jacks ('exercise_detectors/jumping_jack.py')
- Uses:
  - **Arm angle** (average of left/right)
  - **Leg spread** via normalized ankle distance:
    - 'norm_ankle_dist = ankle_dist / base_hip_width'
- **Calibration**: stores 'base_hip_width' for normalization.
- **State machine**:
  - "down" → "up" when arms are raised and legs are apart
  - "up" → "down" when arms lower and legs come together, increments rep count

Key settings:
- 'JUMPING_JACK_ARM_THRESHOLD' (default '86.0')
- 'JUMPING_JACK_LEG_THRESHOLD' (default '1.15')

## Configuration

All core thresholds and MediaPipe confidence values live in:

- 'config/settings.py'

Includes:
- thresholds for each exercise detector
- 'EXERCISE_MODES' mapping
- 'POSE_MIN_DETECTION_CONFIDENCE' (default '0.4')
- 'POSE_MIN_TRACKING_CONFIDENCE' (default '0.4')

## Project Structure

'''
.
├── main.py                       # Main entry point (webcam loop + UI + mode switching)
├── pose_guide.py                 # Calibration instruction overlays + keypoint extraction helpers
├── requirements.txt              # Python dependencies
├── config/
│   └── settings.py               # Thresholds + exercise modes + MediaPipe confidence settings
├── exercise_detectors/
│   ├── __init__.py               # Exposes detector classes
│   ├── base_detector.py          # BaseExerciseDetector interface
│   ├── stationary_running.py
│   ├── pushup.py
│   ├── squat.py
│   └── jumping_jack.py
└── utils/
    ├── angle_utils.py            # Angle computation utility
    └── visualization.py          # Drawing helpers (landmarks, counters, status text)
'''

## Installation

### 1) Clone
'''bash
git clone https://github.com/HTHao-git/exercise-assistance.git
cd exercise-assistance
'''

### 2) Create a virtual environment (recommended)
'''bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
'''

### 3) Install dependencies
'''bash
pip install -r requirements.txt
'''

## Usage

Run the main app:

'''bash
python main.py
'''

### Keyboard Controls

Inside the OpenCV window:
- 'm' — change exercise mode
- 'c' — calibrate (or toggle camera test calibration overlay in mode 0)
- 'd' — toggle debug info (shows detector internals in the right panel)
- 'q' — quit

### Recommended workflow
1. Start in **Camera and Position Testing** (mode '0').
2. Ensure your full body is visible and lighting is good.
3. Press 'm' to select an exercise.
4. Press 'c' and follow the calibration instructions.
5. Start moving and watch the rep counter increase.

## Dependencies

Major dependencies (see 'requirements.txt' for pinned versions):
- 'opencv-python' / 'opencv-contrib-python' — camera input and UI window rendering
- 'mediapipe' — pose landmark detection
- 'numpy' — vector math
- 'scipy', 'matplotlib' — included in requirements (not necessarily required for core runtime)
- 'sounddevice' — included (not used in current main loop code)

## Troubleshooting

### 1) Camera not opening
- Ensure no other app is using the webcam.
- Try changing 'cv2.VideoCapture(0)' to 'cv2.VideoCapture(1)' (different camera index).

### 2) “No landmarks” / poor detection
- Improve lighting, reduce background clutter.
- Step back so your full body fits in frame.
- Keep the camera stable.

### 3) Push-up detector error (possible)
If you see an error like:
- 'AttributeError: 'PushupDetector' object has no attribute 'calculate_angle'

It is likely because 'PushupDetector' calls 'self.calculate_angle(...)' but doesn’t define that method.  
A quick fix is to use 'utils.angle_utils.calculate_angle(...)' like the squat/jumping jack detectors.

## Extending the Project (Add a new exercise)

1. Create a new detector in 'exercise_detectors/' that subclasses 'BaseExerciseDetector'.
2. Implement:
   - 'calibrate(...)' (optional but recommended)
   - 'detect(...)' (rep counting logic + stage transitions)
3. Add it to:
   - 'exercise_detectors/__init__.py'
   - 'config/settings.py' ('EXERCISE_MODES')
   - 'main.py' detector mapping ('self.detectors')

## Safety Disclaimer

This project is for educational/fitness tracking purposes only. It is not medical advice, and pose detection may be inaccurate depending on lighting, camera angle, and user positioning.

## License

No license file is currently included. If you plan to share or reuse this project, consider adding a 'LICENSE' file (e.g., MIT, Apache-2.0).
