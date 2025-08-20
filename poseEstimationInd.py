import cv2
import mediapipe as mp
import time
import math
import json
import os
from collections import deque
from datetime import datetime
import numpy as np

print("Starting hand pose estimation with gesture training integration...")

# -------------------------------
# Training Data Configuration
# -------------------------------
GESTURE_LABELS = {
    '1': 'number_one',
    '2': 'number_two', 
    '3': 'number_three',
    '4': 'number_four',
    '5': 'number_five',
    'm': 'middle_finger',
    'w': 'wave',
    'c': 'come_here',
    'f': 'go_forward',
    'r': 'clockwise',
    'l': 'counter_clockwise',
    's': 'stop',
    'g': 'fight',
    'b': 'grab',
    'o': 'ok_sign',
    't': 'thumbs_up',
    'd': 'thumbs_down',
    'h': 'point',
    'x': 'no_gesture'
}

# Recording parameters
COOLDOWN_DURATION = 3.0  # seconds of cooldown before recording starts
FRAME_RATE = 30

# Training data storage - MODIFIED WORKFLOW
recording_state = 'idle'  # 'idle', 'cooldown', 'recording'
current_gesture = None
recorded_frames = []
cooldown_start_time = None
recording_start_time = None

# Create data directory
DATA_DIR = "gesture_training_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# -------------------------------
# Initialize MediaPipe Hands           
# -------------------------------
print("Initializing MediaPipe Hands...")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)

# -------------------------------
# Hand size estimation constants
# -------------------------------
# Average adult hand measurements (in cm)
AVERAGE_HAND_SPAN = 18.0  # Distance from thumb tip to pinky tip when spread
AVERAGE_HAND_LENGTH = 18.5  # Distance from wrist to middle finger tip
AVERAGE_PALM_WIDTH = 8.5   # Width of palm

# Camera parameters (you may need to adjust these for your camera)
# These are typical values for common webcams
CAMERA_FOV_DEGREES = 60  # Field of view in degrees
FOCAL_LENGTH_PIXELS = 500  # Approximate focal length in pixels (will be estimated)

# -------------------------------
# Calibration system (now using multiple palm edges)
# -------------------------------
CALIBRATION_FRAMES = 60  # Number of frames to collect calibration data
calibrated_palm_measurements = None  # Will store the calibrated palm measurements
calibration_data = []  # Store palm measurements during calibration
calibration_mode = True  # Start in calibration mode
calibration_frame_count = 0

print("CALIBRATION MODE ACTIVE")
print("Please hold your hand naturally in front of the camera")
print("Rotate your hand in different orientations - palm facing camera, turned left/right")
print(f"Collecting data for {CALIBRATION_FRAMES} frames...")
print("The system will measure multiple palm edges and use the most stable one")
print("Press 'r' to restart calibration, 's' to skip calibration")

# -------------------------------
# Multi-edge palm measurement functions
# -------------------------------
def calculate_all_palm_measurements(landmarks, frame_width, frame_height):
    """
    Calculate multiple palm measurements:
    1. Index MCP to Pinky MCP (palm width)
    2. Thumb CMC to Pinky MCP (diagonal width)
    3. Wrist to middle of MCP joints (palm length)
    """
    measurements = {}
    positions = {}
    
    # Get key landmark positions
    wrist = landmarks[0]
    thumb_cmc = landmarks[1]    # Thumb CMC joint
    index_mcp = landmarks[5]    # Index finger MCP joint
    middle_mcp = landmarks[9]   # Middle finger MCP joint
    ring_mcp = landmarks[13]    # Ring finger MCP joint
    pinky_mcp = landmarks[17]   # Pinky finger MCP joint
    
    # Convert to pixel coordinates
    points = {}
    for name, landmark in [
        ('wrist', wrist), ('thumb_cmc', thumb_cmc), ('index_mcp', index_mcp),
        ('middle_mcp', middle_mcp), ('ring_mcp', ring_mcp), ('pinky_mcp', pinky_mcp)
    ]:
        points[name] = (landmark.x * frame_width, landmark.y * frame_height)
    
    # Measurement 1: Index MCP to Pinky MCP (traditional palm width)
    index_to_pinky = math.sqrt(
        (points['index_mcp'][0] - points['pinky_mcp'][0])**2 + 
        (points['index_mcp'][1] - points['pinky_mcp'][1])**2
    )
    measurements['index_to_pinky'] = index_to_pinky
    positions['index_to_pinky'] = (points['index_mcp'], points['pinky_mcp'])
    
    # Measurement 2: Thumb CMC to Pinky MCP (diagonal width)
    thumb_to_pinky = math.sqrt(
        (points['thumb_cmc'][0] - points['pinky_mcp'][0])**2 + 
        (points['thumb_cmc'][1] - points['pinky_mcp'][1])**2
    )
    measurements['thumb_to_pinky'] = thumb_to_pinky
    positions['thumb_to_pinky'] = (points['thumb_cmc'], points['pinky_mcp'])
    
    # Measurement 3: Wrist to MCP center (palm length)
    mcp_center_x = (points['index_mcp'][0] + points['middle_mcp'][0] + 
                    points['ring_mcp'][0] + points['pinky_mcp'][0]) / 4
    mcp_center_y = (points['index_mcp'][1] + points['middle_mcp'][1] + 
                    points['ring_mcp'][1] + points['pinky_mcp'][1]) / 4
    
    wrist_to_mcp_center = math.sqrt(
        (points['wrist'][0] - mcp_center_x)**2 + 
        (points['wrist'][1] - mcp_center_y)**2
    )
    measurements['wrist_to_mcp'] = wrist_to_mcp_center
    positions['wrist_to_mcp'] = (points['wrist'], (mcp_center_x, mcp_center_y))
    
    return measurements, positions

def select_best_measurement(current_measurements, calibrated_measurements):
    """
    Select the measurement that's highest % to calibration value.
    Returns the measurement name, current value, and ratio to calibrated value
    """
    if calibrated_measurements is None:
        # During calibration, return the first measurement
        return 'index_to_pinky', current_measurements['index_to_pinky'], 1.0
    
    best_measurement = None
    best_ratio = None
    highest_ratio = float('-inf')
    
    for measurement_name in current_measurements:
        if measurement_name in calibrated_measurements:
            current_val = current_measurements[measurement_name]
            calibrated_val = calibrated_measurements[measurement_name]
            
            if calibrated_val > 0:  # Avoid division by zero
                ratio = current_val / calibrated_val
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_ratio = ratio
                    best_measurement = measurement_name
    
    if best_measurement is None:
        # Fallback to first measurement
        return 'index_to_pinky', current_measurements['index_to_pinky'], 1.0
    
    return best_measurement, current_measurements[best_measurement], best_ratio

def calculate_palm_orientation(landmarks, frame_width, frame_height):
    """Calculate palm orientation angle to compensate for rotation"""
    # Use three key palm points to define orientation
    wrist = landmarks[0]
    index_mcp = landmarks[5]   # Index MCP
    middle_mcp = landmarks[9]  # Middle MCP
    
    # Convert to pixel coordinates
    wrist_x, wrist_y = wrist.x * frame_width, wrist.y * frame_height
    index_x, index_y = index_mcp.x * frame_width, index_mcp.y * frame_height
    middle_x, middle_y = middle_mcp.x * frame_width, middle_mcp.y * frame_height
    
    # Calculate palm orientation vector (wrist to middle of knuckles)
    palm_center_x = (index_x + middle_x) / 2
    palm_center_y = (index_y + middle_y) / 2
    
    # Vector from wrist to palm center
    palm_vector_x = palm_center_x - wrist_x
    palm_vector_y = palm_center_y - wrist_y
    
    # Calculate angle (in degrees) - 0° when palm points up
    angle_radians = math.atan2(palm_vector_x, -palm_vector_y)  # -y because screen coords are flipped
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def calculate_hand_length_pixels(landmarks, frame_width, frame_height):
    """Calculate the pixel distance from wrist to middle finger tip"""
    wrist = landmarks[0]  # Wrist
    middle_tip = landmarks[12]  # Middle finger tip
    
    wrist_x = wrist.x * frame_width
    wrist_y = wrist.y * frame_height
    middle_x = middle_tip.x * frame_width
    middle_y = middle_tip.y * frame_height
    
    # Calculate Euclidean distance in pixels
    length_pixels = math.sqrt((wrist_x - middle_x)**2 + (wrist_y - middle_y)**2)
    return length_pixels, (wrist_x, wrist_y), (middle_x, middle_y)

def estimate_distance_from_multi_edge_palm(current_measurements, calibrated_measurements, frame_width, assumed_focal_length=500):
    """
    Estimate distance using the best palm measurement as reference
    """
    if calibrated_measurements is None:
        return None, None, None, None
    
    best_measurement_name, best_current_value, ratio = select_best_measurement(
        current_measurements, calibrated_measurements)
    
    if best_current_value < 5:  # Too small to be reliable
        return None, None, None, None
    
    # The calibrated measurement represents our "reference distance" 
    reference_distance_cm = 40.0
    
    # Calculate current distance using the best measurement
    calibrated_value = calibrated_measurements[best_measurement_name]
    distance_cm = (calibrated_value * reference_distance_cm) / best_current_value
    
    return distance_cm, best_measurement_name, best_current_value, ratio

def calculate_palm_plane_reference(landmarks, frame_width, frame_height):
    """
    Calculate a reference palm plane using key palm points
    Returns the average Z-coordinate of palm landmarks
    """
    # Use stable palm landmarks to define the palm plane
    palm_landmarks = [0, 1, 5, 9, 13, 17]  # Wrist, Thumb CMC, and all MCP joints
    
    palm_z_sum = 0
    for landmark_idx in palm_landmarks:
        palm_z_sum += landmarks[landmark_idx].z
    
    palm_reference_z = palm_z_sum / len(palm_landmarks)
    return palm_reference_z

def calculate_enhanced_fingertip_depth(landmarks, fingertip_idx, hand_distance_cm, frame_width, frame_height):
    """
    Calculate fingertip depth using multiple methods combined
    """
    if hand_distance_cm is None:
        return None
    
    fingertip = landmarks[fingertip_idx]
    
    # Method 1: Enhanced MediaPipe Z scaling
    palm_reference_z = calculate_palm_plane_reference(landmarks, frame_width, frame_height)
    
    # Calculate relative depth from palm plane
    relative_z = fingertip.z - palm_reference_z
    
    # Scale the relative Z based on hand size (bigger hands = bigger Z variations)
    # Use hand distance as a scaling factor
    z_scale_factor = hand_distance_cm * 0.3  # Adjust this multiplier as needed
    scaled_relative_z = relative_z * z_scale_factor
    
    # Method 2: Finger length analysis
    finger_base_idx = None
    expected_finger_length_cm = 0
    
    # Map fingertip to its base joint and expected length
    if fingertip_idx == 4:  # Thumb tip
        finger_base_idx = 2  # Thumb MCP
        expected_finger_length_cm = 5.5
    elif fingertip_idx == 8:  # Index tip
        finger_base_idx = 5  # Index MCP
        expected_finger_length_cm = 7.5
    elif fingertip_idx == 12:  # Middle tip
        finger_base_idx = 9  # Middle MCP
        expected_finger_length_cm = 8.5
    elif fingertip_idx == 16:  # Ring tip
        finger_base_idx = 13  # Ring MCP
        expected_finger_length_cm = 7.8
    elif fingertip_idx == 20:  # Pinky tip
        finger_base_idx = 17  # Pinky MCP
        expected_finger_length_cm = 6.0
    
    depth_from_finger_analysis = 0
    if finger_base_idx is not None:
        # Calculate apparent finger length in pixels
        base_joint = landmarks[finger_base_idx]
        
        base_x = base_joint.x * frame_width
        base_y = base_joint.y * frame_height
        tip_x = fingertip.x * frame_width
        tip_y = fingertip.y * frame_height
        
        apparent_length_pixels = math.sqrt((tip_x - base_x)**2 + (tip_y - base_y)**2)
        
        # Convert to real-world length
        apparent_length_cm = (apparent_length_pixels * hand_distance_cm) / 500  # Using focal length
        
        # If finger appears shorter than expected, it might be pointing toward camera
        if apparent_length_cm < expected_finger_length_cm * 0.8:  # 20% tolerance
            length_ratio = apparent_length_cm / expected_finger_length_cm
            # Estimate how much the finger is pointing forward
            # Using Pythagorean theorem: if finger appears shorter, the "missing" length is depth
            if length_ratio > 0.3:  # Avoid extreme values
                estimated_forward_depth = expected_finger_length_cm * math.sqrt(1 - length_ratio**2)
                depth_from_finger_analysis = -estimated_forward_depth  # Negative = closer to camera
    
    # Method 3: Fingertip-to-wrist distance analysis
    wrist = landmarks[0]
    wrist_x = wrist.x * frame_width
    wrist_y = wrist.y * frame_height
    tip_x = fingertip.x * frame_width
    tip_y = fingertip.y * frame_height
    
    # Calculate 2D distance from wrist to fingertip
    wrist_to_tip_2d = math.sqrt((tip_x - wrist_x)**2 + (tip_y - wrist_y)**2)
    wrist_to_tip_2d_cm = (wrist_to_tip_2d * hand_distance_cm) / 500
    
    # Expected 3D distance from wrist to fingertip (typical hand span)
    expected_wrist_to_tip_3d = 18.0  # cm, typical hand span
    
    # If 2D distance is significantly less than expected 3D, finger might be pointing forward
    if wrist_to_tip_2d_cm < expected_wrist_to_tip_3d * 0.7:
        depth_from_wrist_analysis = -math.sqrt(expected_wrist_to_tip_3d**2 - wrist_to_tip_2d_cm**2) * 0.5
    else:
        depth_from_wrist_analysis = 0
    
    # Combine methods with weights
    method1_weight = 0.4  # MediaPipe Z scaling
    method2_weight = 0.4  # Finger length analysis
    method3_weight = 0.2  # Wrist distance analysis
    
    combined_depth_offset = (
        scaled_relative_z * method1_weight +
        depth_from_finger_analysis * method2_weight +
        depth_from_wrist_analysis * method3_weight
    )
    
    # Final depth calculation
    final_depth = hand_distance_cm + combined_depth_offset
    
    return final_depth, {
        'palm_reference_z': palm_reference_z,
        'relative_z': relative_z,
        'scaled_relative_z': scaled_relative_z,
        'finger_analysis': depth_from_finger_analysis,
        'wrist_analysis': depth_from_wrist_analysis,
        'combined_offset': combined_depth_offset
    }

def calculate_fingertip_3d_position(landmark, fingertip_idx, landmarks, hand_distance_cm, frame_width, frame_height, focal_length=500):
    """
    Calculate the 3D position of a fingertip in real-world coordinates with enhanced depth
    Returns (x_cm, y_cm, z_cm) where z is distance from camera
    """
    if hand_distance_cm is None:
        return None, None, None, None
    
    # Convert normalized coordinates to pixel coordinates
    pixel_x = landmark.x * frame_width
    pixel_y = landmark.y * frame_height
    
    # Convert pixel coordinates to real-world coordinates
    # Center the coordinates (0,0 at center of image)
    centered_x = pixel_x - (frame_width / 2)
    centered_y = pixel_y - (frame_height / 2)
    
    # Calculate real-world x,y coordinates using similar triangles
    real_x_cm = (centered_x * hand_distance_cm) / focal_length
    real_y_cm = (centered_y * hand_distance_cm) / focal_length
    
    # Enhanced depth calculation
    real_z_cm, depth_debug = calculate_enhanced_fingertip_depth(
        landmarks, fingertip_idx, hand_distance_cm, frame_width, frame_height)
    
    return real_x_cm, real_y_cm, real_z_cm, depth_debug

# -------------------------------
# Training data extraction functions
# -------------------------------
def extract_comprehensive_hand_features(hand_landmarks, handedness, hand_distance_cm, frame_width, frame_height, focal_length=500):
    """Extract comprehensive features including enhanced 3D positions and motion data"""
    features = {
        'hand_label': handedness.classification[0].label,
        'hand_confidence': handedness.classification[0].score,
        'hand_distance_cm': hand_distance_cm,
        'palm_measurements': {},
        'landmarks_3d': [],  # Real-world coordinates
        'landmarks_normalized': [],  # Original MediaPipe coordinates
        'fingertip_positions': {},
        'palm_orientation': 0,
        'finger_angles': [],
        'distances': []
    }
    
    landmarks = hand_landmarks.landmark
    
    # Calculate palm measurements and orientation
    current_measurements, _ = calculate_all_palm_measurements(landmarks, frame_width, frame_height)
    features['palm_measurements'] = current_measurements
    features['palm_orientation'] = calculate_palm_orientation(landmarks, frame_width, frame_height)
    
    # Extract all landmark positions (both normalized and real-world)
    for i, landmark in enumerate(landmarks):
        # Normalized coordinates (MediaPipe default)
        features['landmarks_normalized'].extend([landmark.x, landmark.y, landmark.z])
        
        # Real-world coordinates if distance is available
        if hand_distance_cm is not None:
            pixel_x = landmark.x * frame_width
            pixel_y = landmark.y * frame_height
            centered_x = pixel_x - (frame_width / 2)
            centered_y = pixel_y - (frame_height / 2)
            real_x_cm = (centered_x * hand_distance_cm) / focal_length
            real_y_cm = (centered_y * hand_distance_cm) / focal_length
            
            # Use enhanced depth for fingertips, basic scaling for others
            if i in [4, 8, 12, 16, 20]:  # Fingertips
                real_z_cm, _ = calculate_enhanced_fingertip_depth(landmarks, i, hand_distance_cm, frame_width, frame_height)
                if real_z_cm is None:
                    real_z_cm = hand_distance_cm
            else:
                # Basic depth scaling for non-fingertip landmarks
                palm_ref_z = calculate_palm_plane_reference(landmarks, frame_width, frame_height)
                relative_z = landmark.z - palm_ref_z
                real_z_cm = hand_distance_cm + (relative_z * hand_distance_cm * 0.2)
            
            features['landmarks_3d'].extend([real_x_cm, real_y_cm, real_z_cm])
        else:
            # Fallback to normalized if no distance available
            features['landmarks_3d'].extend([landmark.x, landmark.y, landmark.z])
    
    # Extract fingertip positions with enhanced depth
    fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    fingertip_names = ["thumb", "index", "middle", "ring", "pinky"]
    
    for tip_idx, name in zip(fingertips, fingertip_names):
        real_x, real_y, real_z, depth_debug = calculate_fingertip_3d_position(
            landmarks[tip_idx], tip_idx, landmarks, hand_distance_cm, frame_width, frame_height, focal_length)
        if real_x is not None:
            features['fingertip_positions'][name] = {
                'x': real_x, 'y': real_y, 'z': real_z,
                'depth_debug': depth_debug
            }
    
    # Calculate finger bend angles
    finger_joints = [
        [1, 2, 3, 4],      # Thumb
        [5, 6, 7, 8],      # Index
        [9, 10, 11, 12],   # Middle
        [13, 14, 15, 16],  # Ring
        [17, 18, 19, 20]   # Pinky
    ]
    
    for joints in finger_joints:
        if len(joints) >= 3:
            p1 = landmarks[joints[0]]
            p2 = landmarks[joints[1]] 
            p3 = landmarks[joints[2]]
            
            # Calculate angle at middle joint
            v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]
            v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z]
            
            dot_product = sum(a*b for a, b in zip(v1, v2))
            mag1 = math.sqrt(sum(a*a for a in v1))
            mag2 = math.sqrt(sum(a*a for a in v2))
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = math.acos(cos_angle)
                features['finger_angles'].append(angle)
            else:
                features['finger_angles'].append(0)
    
    # Calculate key distances (3D)
    palm_center = landmarks[0]  # Wrist as palm reference
    for tip_idx in fingertips:
        tip = landmarks[tip_idx]
        distance = math.sqrt(
            (tip.x - palm_center.x)**2 + 
            (tip.y - palm_center.y)**2 + 
            (tip.z - palm_center.z)**2
        )
        features['distances'].append(distance)
    
    return features

def calculate_motion_features(current_features, previous_features, dt):
    """Calculate motion features between frames"""
    motion_features = {
        'palm_velocity': 0,
        'palm_acceleration': 0,
        'fingertip_velocities': {},
        'overall_motion_magnitude': 0,
        'motion_direction': [0, 0, 0]
    }
    
    if previous_features is None or dt <= 0:
        return motion_features
    
    # Calculate palm motion (using wrist position)
    if 'landmarks_3d' in current_features and 'landmarks_3d' in previous_features:
        if len(current_features['landmarks_3d']) >= 3 and len(previous_features['landmarks_3d']) >= 3:
            # Wrist is first landmark (indices 0, 1, 2 for x, y, z)
            current_wrist = current_features['landmarks_3d'][0:3]
            previous_wrist = previous_features['landmarks_3d'][0:3]
            
            velocity_vector = [(curr - prev) / dt for curr, prev in zip(current_wrist, previous_wrist)]
            motion_features['palm_velocity'] = math.sqrt(sum(v*v for v in velocity_vector))
            motion_features['motion_direction'] = velocity_vector
    
    # Calculate fingertip velocities
    for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        if (finger_name in current_features['fingertip_positions'] and 
            finger_name in previous_features['fingertip_positions']):
            
            curr_pos = current_features['fingertip_positions'][finger_name]
            prev_pos = previous_features['fingertip_positions'][finger_name]
            
            dx = (curr_pos['x'] - prev_pos['x']) / dt
            dy = (curr_pos['y'] - prev_pos['y']) / dt
            dz = (curr_pos['z'] - prev_pos['z']) / dt
            
            velocity = math.sqrt(dx*dx + dy*dy + dz*dz)
            motion_features['fingertip_velocities'][finger_name] = {
                'velocity': velocity,
                'velocity_vector': [dx, dy, dz]
            }
    
    # Calculate overall motion magnitude
    velocities = [motion_features['palm_velocity']]
    for finger_data in motion_features['fingertip_velocities'].values():
        velocities.append(finger_data['velocity'])
    motion_features['overall_motion_magnitude'] = sum(velocities) / len(velocities) if velocities else 0
    
    return motion_features

def save_training_sample(frames_data, gesture_label):
    """Save a training sample to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{DATA_DIR}/{gesture_label}_{timestamp}.json"
    
    sample = {
        'label': gesture_label,
        'timestamp': timestamp,
        'duration': len(frames_data) / FRAME_RATE,  # Actual duration based on frames
        'frame_count': len(frames_data),
        'sequence': frames_data
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(sample, f, indent=2)
        print(f"Saved training sample: {filename}")
        return True
    except Exception as e:
        print(f"Error saving sample: {e}")
        return False

def draw_training_ui(frame, recording_state, current_gesture, recorded_frames, cooldown_start_time, recording_start_time, current_time):
    """Draw training UI elements for the new workflow"""
    frame_height, frame_width = frame.shape[:2]
    
    if recording_state == 'cooldown':
        # Cooldown countdown
        elapsed_cooldown = current_time - cooldown_start_time
        remaining_cooldown = max(0, COOLDOWN_DURATION - elapsed_cooldown)
        
        # Large countdown display
        countdown_text = f"GET READY: {remaining_cooldown:.1f}"
        font_scale = 1.5
        text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height) // 2
        
        # Background rectangle for visibility
        cv2.rectangle(frame, (text_x - 20, text_y - 40), (text_x + text_size[0] + 20, text_y + 20), (0, 0, 0), -1)
        cv2.putText(frame, countdown_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 3)
        
        # Gesture name
        gesture_text = f"Gesture: {current_gesture.replace('_', ' ').upper()}"
        gesture_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        gesture_x = (frame_width - gesture_size[0]) // 2
        cv2.putText(frame, gesture_text, (gesture_x, text_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Progress bar
        progress = elapsed_cooldown / COOLDOWN_DURATION
        bar_width = 400
        bar_height = 20
        bar_x = (frame_width - bar_width) // 2
        bar_y = text_y + 40
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 255), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
    elif recording_state == 'recording':
        # Recording indicator
        elapsed_recording = current_time - recording_start_time
        
        # Large recording display
        recording_text = "RECORDING"
        font_scale = 1.2
        text_size = cv2.getTextSize(recording_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = 80
        
        # Flashing red background
        flash_intensity = int(abs(math.sin(current_time * 4)) * 255)  # Flash at 4Hz
        cv2.rectangle(frame, (text_x - 20, text_y - 40), (text_x + text_size[0] + 20, text_y + 20), (0, 0, flash_intensity), -1)
        cv2.putText(frame, recording_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
        
        # Gesture name and frame count
        gesture_text = f"{current_gesture.replace('_', ' ').upper()} - Frame {len(recorded_frames)}"
        gesture_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        gesture_x = (frame_width - gesture_size[0]) // 2
        cv2.putText(frame, gesture_text, (gesture_x, text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Recording duration
        duration_text = f"Duration: {elapsed_recording:.1f}s"
        duration_size = cv2.getTextSize(duration_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        duration_x = (frame_width - duration_size[0]) // 2
        cv2.putText(frame, duration_text, (duration_x, text_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Instructions
        instruction_text = "Press SPACEBAR to stop and save"
        instruction_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        instruction_x = (frame_width - instruction_size[0]) // 2
        cv2.putText(frame, instruction_text, (instruction_x, text_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    else:  # idle state
        # Gesture legend (smaller, right side)
        legend_x = frame_width - 300
        legend_y = 120
        cv2.putText(frame, "Gesture Keys (Press to start):", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        legend_items = [
            "1-4: Numbers", "M: Middle finger", "W: Wave", 
            "C: Come here", "F: Go forward", "S: Stop", "G: Fight", "B: Grab",
            "O: OK", "T: Thumbs up", "D: Thumbs down",
            "H: Point", "X: No gesture"
        ]
        
        for i, item in enumerate(legend_items):
            if i < 12:  # Limit display to prevent overflow
                cv2.putText(frame, item, (legend_x, legend_y + 25 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Instructions
        instruction_y = frame_height - 100
        cv2.putText(frame, "NEW WORKFLOW:", (10, instruction_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "1. Press gesture key → 3s cooldown → Recording starts", (10, instruction_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "2. Perform gesture → Press SPACEBAR to stop and save", (10, instruction_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def print_gesture_instructions():
    """Print gesture training instructions"""
    print("\n" + "="*70)
    print("GESTURE TRAINING MODE - NEW WORKFLOW")
    print("="*70)
    print("Instructions:")
    print("1. Complete calibration first (or press 's' to skip)")
    print("2. Position your hand in view of the camera")
    print("3. Press a gesture key to start 3-second cooldown")
    print("4. Get ready during cooldown period")
    print("5. Recording starts automatically after cooldown")
    print("6. Perform the gesture naturally")
    print("7. Press SPACEBAR when done to stop and save")
    print("\nGesture Keys:")
    for key, label in GESTURE_LABELS.items():
        print(f"  {key.upper()}: {label.replace('_', ' ').title()}")
    print("\nNote: Motion gestures like 'wave', 'go_forward', 'clockwise'")
    print("      should include actual movement while recording")
    print("      Recording duration is flexible - you control when to stop!")
    print("="*70)

# -------------------------------
# Configuration
# -------------------------------
FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
FINGER_NAMES = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}
Y_THRESHOLD = 300

# Distance threshold for detection (in cm from camera)
DISTANCE_THRESHOLD = 30.0  # Alert when fingertips are closer than 30cm

# -------------------------------
# Open webcam
# -------------------------------
print("Opening webcam...")
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    input("Press Enter to exit...")
    exit(1)

# Test webcam
ret, test_frame = cap.read()
if not ret or test_frame is None:
    print("ERROR: Could not read from webcam!")
    cap.release()
    input("Press Enter to exit...")
    exit(1)

frame_height, frame_width = test_frame.shape[:2]
print(f"Webcam working! Frame size: {frame_width}x{frame_height}")
print(f"Distance threshold: {DISTANCE_THRESHOLD}cm")
print("Controls: Press 'q' or ESC to quit")
print("-" * 50)

# Print gesture training instructions
print_gesture_instructions()

# FPS tracking
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0
frame_count = 0

# Calibration tracking
distance_measurements = []
focal_length_estimate = FOCAL_LENGTH_PIXELS

# Motion tracking for gestures
previous_hand_features = None
last_frame_time = time.time()

print("Starting main loop...")

try:
    finger_states = {tip_idx: False for tip_idx in FINGER_TIPS}  # False = below threshold
    fingstream = {}
    for findx in FINGER_TIPS:
        fingstream[findx] = deque()

    log_backlog = []  # Store events until thumb triggers flush
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        frame_count += 1
        current_time = time.time()
        dt = current_time - last_frame_time
        last_frame_time = current_time
        
        # Calculate FPS
        fps_frame_count += 1
        if fps_frame_count >= 30:
            fps_end_time = time.time()
            current_fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            fps_frame_count = 0

        # -------------------------------
        # Handle recording state transitions
        # -------------------------------
        if recording_state == 'cooldown':
            elapsed_cooldown = current_time - cooldown_start_time
            if elapsed_cooldown >= COOLDOWN_DURATION:
                # Start recording
                recording_state = 'recording'
                recording_start_time = current_time
                recorded_frames = []
                print(f"Recording started for gesture: {current_gesture}")
                if current_gesture in ['wave', 'go_forward', 'come_here', 'clockwise', 'counter_clockwise']:
                    print("*** MOTION GESTURE: Move your hand during recording! ***")
                else:
                    print("Perform the gesture now! Press SPACEBAR when done.")

        try:
            # -------------------------------
            # Hand detection
            # -------------------------------
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            hands_detected = 0
            current_frame_training_data = []
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hands_detected = len(results.multi_hand_landmarks)
                
                for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    hand_label = handedness.classification[0].label
                    
                    # Draw hand connections
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Calculate all palm measurements
                    current_measurements, measurement_positions = calculate_all_palm_measurements(
                        hand_landmarks.landmark, frame_width, frame_height)
                    
                    palm_angle = calculate_palm_orientation(hand_landmarks.landmark, frame_width, frame_height)
                    
                    # -------------------------------
                    # CALIBRATION MODE
                    # -------------------------------
                    if calibration_mode:
                        calibration_frame_count += 1
                        calibration_data.append(current_measurements.copy())
                        
                        # Draw calibration progress
                        progress = calibration_frame_count / CALIBRATION_FRAMES
                        cv2.rectangle(frame, (50, 50), (50 + int(300 * progress), 80), (0, 255, 0), -1)
                        cv2.rectangle(frame, (50, 50), (350, 80), (255, 255, 255), 2)
                        
                        calibration_text = f"CALIBRATING... {calibration_frame_count}/{CALIBRATION_FRAMES}"
                        cv2.putText(frame, calibration_text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(frame, "Rotate hand naturally!", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Show current measurements during calibration
                        y_offset = 170
                        for i, (measurement_name, value) in enumerate(current_measurements.items()):
                            display_name = measurement_name.replace('_', ' ').title()
                            cv2.putText(frame, f"{display_name}: {value:.1f}px", 
                                       (50, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.putText(frame, f"Palm angle: {palm_angle:.1f}°", (50, y_offset + len(current_measurements) * 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Draw all measurement lines during calibration
                        colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255)]  # Cyan, Yellow, Magenta
                        for i, (measurement_name, positions) in enumerate(measurement_positions.items()):
                            color = colors[i % len(colors)]
                            start_pos, end_pos = positions
                            cv2.line(frame, (int(start_pos[0]), int(start_pos[1])), 
                                    (int(end_pos[0]), int(end_pos[1])), color, 2)
                        
                        if calibration_frame_count >= CALIBRATION_FRAMES:
                            # Calibration complete - calculate average of all measurements
                            calibrated_palm_measurements = {}
                            for measurement_name in current_measurements.keys():
                                values = [frame_data[measurement_name] for frame_data in calibration_data]
                                calibrated_palm_measurements[measurement_name] = sum(values) / len(values)
                            
                            calibration_mode = False
                            print(f"\nCALIBRATION COMPLETE!")
                            print("Calibrated palm measurements:")
                            for measurement_name, value in calibrated_palm_measurements.items():
                                print(f"  {measurement_name.replace('_', ' ').title()}: {value:.1f} pixels")
                            print("System will now select the most stable measurement for each frame")
                            print("Press 'r' to recalibrate if needed")
                            print("\n*** GESTURE TRAINING NOW AVAILABLE ***")
                            print("Press gesture keys to start recording workflow!")
                    
                    # -------------------------------
                    # NORMAL OPERATION MODE
                    # -------------------------------
                    else:
                        # Calculate distance using best palm measurement
                        hand_distance, best_measurement, best_value, ratio = estimate_distance_from_multi_edge_palm(
                            current_measurements, calibrated_palm_measurements, frame_width, focal_length_estimate
                        )
                        
                        if hand_distance is not None:
                            # Extract comprehensive features for training
                            hand_features = extract_comprehensive_hand_features(
                                hand_landmarks, handedness, hand_distance, frame_width, frame_height, focal_length_estimate)
                            
                            # Calculate motion features
                            motion_features = calculate_motion_features(hand_features, previous_hand_features, dt)
                            hand_features['motion'] = motion_features
                            hand_features['timestamp'] = current_time
                            hand_features['dt'] = dt
                            
                            # Store for training data if recording
                            current_frame_training_data.append(hand_features)
                            
                            # Update previous features for next frame
                            previous_hand_features = hand_features
                            
                            # Draw the best measurement line (highlighted)
                            if best_measurement in measurement_positions:
                                start_pos, end_pos = measurement_positions[best_measurement]
                                cv2.line(frame, (int(start_pos[0]), int(start_pos[1])), 
                                        (int(end_pos[0]), int(end_pos[1])), (0, 255, 0), 3)  # Green for best
                            
                            # Draw other measurements (dimmed)
                            colors = [(100, 100, 255), (100, 255, 255), (255, 100, 255)]
                            for i, (measurement_name, positions) in enumerate(measurement_positions.items()):
                                if measurement_name != best_measurement:
                                    color = colors[i % len(colors)]
                                    start_pos, end_pos = positions
                                    cv2.line(frame, (int(start_pos[0]), int(start_pos[1])), 
                                            (int(end_pos[0]), int(end_pos[1])), color, 1)
                            
                            # Display hand distance and measurements
                            distance_text = f"{hand_label}: {hand_distance:.1f}cm"
                            cv2.putText(frame, distance_text, (10, 150 + hand_idx * 120), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                            
                            # Show motion info during recording
                            if recording_state == 'recording':
                                motion_text = f"Motion: {motion_features['overall_motion_magnitude']:.2f} cm/s"
                                cv2.putText(frame, motion_text, (10, 175 + hand_idx * 120), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            elif recording_state == 'idle':
                                # Show which measurement is being used
                                best_measurement_display = best_measurement.replace('_', ' ').title()
                                measurement_info = f"Using: {best_measurement_display} ({ratio*100:.0f}% of cal)"
                                cv2.putText(frame, measurement_info, (10, 175 + hand_idx * 120), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            # Show all current measurements (only if idle to avoid clutter)
                            if recording_state == 'idle':
                                y_offset = 195 + hand_idx * 120
                                for i, (measurement_name, value) in enumerate(current_measurements.items()):
                                    display_name = measurement_name.replace('_', ' ').title()
                                    calibrated_value = calibrated_palm_measurements.get(measurement_name, 0)
                                    current_ratio = (value / calibrated_value * 100) if calibrated_value > 0 else 0
                                    color = (0, 255, 0) if measurement_name == best_measurement else (150, 150, 150)
                                    cv2.putText(frame, f"{display_name}: {current_ratio:.0f}%", 
                                                (10, y_offset + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                
                                angle_info = f"Angle: {palm_angle:.1f}°"
                                cv2.putText(frame, angle_info, (10, y_offset + len(current_measurements) * 15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                            
                            # Process each fingertip (original functionality) - only in idle state
                            if recording_state == 'idle':
                                fingertip_positions = []
                                for tip_idx in FINGER_TIPS:
                                    lm = hand_landmarks.landmark[tip_idx]
                                    x_px = int(lm.x * frame_width)
                                    y_px = int(lm.y * frame_height)

                                    real_x, real_y, real_z, depth_debug = calculate_fingertip_3d_position(
                                        lm, tip_idx, hand_landmarks.landmark, hand_distance, frame_width, frame_height, focal_length_estimate
                                    )

                                    if real_z is None:
                                        continue
                                    
                                    # Check thresholds (original functionality)
                                    below_y_threshold = y_px >= Y_THRESHOLD
                                    close_to_camera = real_z < DISTANCE_THRESHOLD
                                    crossed = below_y_threshold or close_to_camera

                                    finger_key = f"{hand_label} {FINGER_NAMES[tip_idx]}"
                                    was_below_threshold = finger_states.get(finger_key, False)

                                    # --- Event logging: only log when crossing threshold ---
                                    if crossed and not was_below_threshold:
                                        log_backlog.append((hand_label, FINGER_NAMES[tip_idx], real_x, real_y, real_z))
                                    # Update finger state
                                    finger_states[finger_key] = crossed

                                    # Color coding for CV display
                                    if close_to_camera and below_y_threshold:
                                        color = (0, 0, 255)  # Red
                                    elif close_to_camera:
                                        color = (0, 165, 255)  # Orange
                                    elif below_y_threshold:
                                        color = (0, 255, 255)  # Yellow
                                    else:
                                        color = (0, 255, 0)  # Green

                                    cv2.circle(frame, (x_px, y_px), 6, color, -1)
                                    cv2.putText(frame, f"{real_z:.1f}cm", (x_px + 8, y_px + 15),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                                # --- Flush logs if any thumb crosses ---
                                for tip_idx in FINGER_TIPS:
                                    if FINGER_NAMES[tip_idx] == 'Thumb' and finger_states.get(f"{hand_label} Thumb", False):
                                        if log_backlog:
                                            with open("finger_logs.txt", "a") as f:
                                                f.write("\n--- New Entry ---\n")
                                                for entry in log_backlog:
                                                    f.write(f"{entry[0]} {entry[1]}: ({entry[2]:.1f}, {entry[3]:.1f}, {entry[4]:.1f})\n")
                                            print(f"Flushed {len(log_backlog)} log entries")
                                            log_backlog.clear()
                                        break  # flush only once per frame

            # -------------------------------
            # Handle gesture recording (NEW WORKFLOW)
            # -------------------------------
            if recording_state == 'recording' and current_frame_training_data:
                recorded_frames.append({
                    'frame_number': len(recorded_frames),
                    'timestamp': current_time,
                    'dt': dt,
                    'hands': current_frame_training_data
                })

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")

        # -------------------------------
        # Draw UI elements
        # -------------------------------
        
        # Draw training UI (NEW)
        draw_training_ui(frame, recording_state, current_gesture, recorded_frames, 
                        cooldown_start_time, recording_start_time, current_time)
        
        # Draw original UI elements (only in idle state)
        if not calibration_mode and recording_state == 'idle':
            cv2.line(frame, (0, Y_THRESHOLD), (frame_width, Y_THRESHOLD), (0, 0, 255), 2)
            cv2.putText(frame, f"Y Threshold: {Y_THRESHOLD}", (frame_width - 200, Y_THRESHOLD - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.putText(frame, f"Distance Threshold: {DISTANCE_THRESHOLD}cm", (10, frame_height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        
        # Status info
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if calibration_mode:
            cv2.putText(frame, f"Hands: {hands_detected}", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"Hands: {hands_detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if recording_state == 'idle':
                cv2.putText(frame, "Multi-Edge Palm Tracking + Gesture Training", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, "Green=Best measurement, Dimmed=Others", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if recording_state == 'idle' and not calibration_mode:
            cv2.putText(frame, "Green=Normal, Yellow=Y-thresh, Orange=Close, Red=Both", 
                       (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display frame
        cv2.imshow("Hand Tracking with Gesture Training - New Workflow", frame)

        # -------------------------------
        # Handle keyboard input (MODIFIED)
        # -------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('r'):  # 'r' for restart calibration
            print("\nRestarting calibration...")
            calibration_mode = True
            calibration_frame_count = 0
            calibration_data = []
            calibrated_palm_measurements = None
            recording_state = 'idle'  # Reset recording state
            current_gesture = None
            recorded_frames = []
        elif key == ord('s') and calibration_mode:  # 's' to skip calibration
            print("\nSkipping calibration - using default values")
            calibrated_palm_measurements = {
                'index_to_pinky': 80,
                'thumb_to_pinky': 120,
                'wrist_to_mcp': 90
            }
            calibration_mode = False
            print("*** GESTURE TRAINING NOW AVAILABLE ***")
            print("Press gesture keys to start recording workflow!")
        elif key == 32 and recording_state == 'recording':  # SPACEBAR to stop recording
            # Stop recording and save
            print(f"Stopping recording for gesture: {current_gesture}")
            if len(recorded_frames) > 0:
                success = save_training_sample(recorded_frames, current_gesture)
                if success:
                    duration = len(recorded_frames) / FRAME_RATE
                    print(f"Successfully saved {len(recorded_frames)} frames ({duration:.2f}s)")
                    # Show motion summary
                    if recorded_frames:
                        total_motion = sum(
                            frame['hands'][0]['motion']['overall_motion_magnitude'] 
                            for frame in recorded_frames 
                            if frame['hands'] and 'motion' in frame['hands'][0]
                        )
                        avg_motion = total_motion / len(recorded_frames) if recorded_frames else 0
                        print(f"Average motion magnitude: {avg_motion:.2f} cm/s")
                else:
                    print("Failed to save recording")
            else:
                print("No frames recorded - nothing to save")
            
            # Reset to idle state
            recording_state = 'idle'
            current_gesture = None
            recorded_frames = []
            recording_start_time = None
            cooldown_start_time = None
        elif key != 255 and recording_state == 'idle' and not calibration_mode:  # Gesture key pressed
            key_char = chr(key).lower()
            if key_char in GESTURE_LABELS:
                # Start cooldown
                current_gesture = GESTURE_LABELS[key_char]
                recording_state = 'cooldown'
                cooldown_start_time = current_time
                recorded_frames = []
                print(f"Starting cooldown for gesture: {current_gesture}")
                print("Get ready! Recording will start automatically after cooldown.")

    print("Exited main loop normally")

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Cleaning up...")
    
    # Create summary of collected data
    if os.path.exists(DATA_DIR):
        try:
            summary = {
                'total_samples': 0,
                'samples_per_gesture': {},
                'gestures': list(GESTURE_LABELS.values()),
                'recording_settings': {
                    'cooldown_duration': COOLDOWN_DURATION,
                    'frame_rate': FRAME_RATE,
                    'workflow': 'flexible_duration_spacebar_stop'
                }
            }
            
            # Count existing files
            for filename in os.listdir(DATA_DIR):
                if filename.endswith('.json') and filename != 'summary.json':
                    gesture = filename.split('_')[0]
                    if gesture not in summary['samples_per_gesture']:
                        summary['samples_per_gesture'][gesture] = 0
                    summary['samples_per_gesture'][gesture] += 1
                    summary['total_samples'] += 1
            
            summary_file = f"{DATA_DIR}/summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nGesture Training Summary:")
            print(f"Total samples collected: {summary['total_samples']}")
            print("Samples per gesture:")
            for gesture, count in summary['samples_per_gesture'].items():
                print(f"  {gesture}: {count}")
            print(f"Data saved in: {DATA_DIR}/")
            
        except Exception as e:
            print(f"Error creating summary: {e}")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Done!")

print("Program ended.")
input("Press Enter to close console...")