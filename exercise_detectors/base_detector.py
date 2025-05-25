class BaseExerciseDetector:
    """Base class for all exercise detectors"""
    
    def __init__(self, name):
        self.name = name
        self.counter = 0
        self.stage = None
        self.calibrated = False
        self.calibration_data = {}
        
    def detect(self, landmarks):
        """
        Detect exercise and return counter and stage
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            tuple: (counter, stage)
        """
        raise NotImplementedError("Subclasses must implement detect()")
        
    def reset(self):
        """Reset the counter and stage"""
        self.counter = 0
        self.stage = None
        self.calibrated = False
        self.calibration_data = {}
        
    def calibrate(self, landmarks, key_points=None):
        """
        Calibrate the detector with user's specific body proportions
        
        Args:
            landmarks: MediaPipe pose landmarks
            key_points: Optional dictionary with extracted key points for the exercise
            
        Returns:
            bool: True if calibration successful, False otherwise
        """
        # Store key points in calibration data if provided
        if key_points:
            self.calibration_data.update(key_points)
            
        # Default implementation - subclasses can override for more specific calibration
        self.calibrated = True
        return True