import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate angle between three points
    Args:
        a: First point [x, y]
        b: Mid point [x, y] (vertex)
        c: End point [x, y]
    Returns:
        Angle in degrees
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    # Calculate vectors
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Check if angle is greater than 180 degrees
    if angle > 180.0:
        angle = 360 - angle
        
    return angle