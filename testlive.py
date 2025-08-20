import cv2
import mediapipe as mp
import time
import math
import json
import os
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
from datetime import datetime

print("Starting Live Gesture Recognition...")

# -------------------------------
# Model Loading Configuration
# -------------------------------
MODEL_PATH = "models/20250818_234215/final_model.h5"
SCALER_PATH = "gesture_scaler.pkl"
LABEL_ENCODER_PATH = "gesture_label_encoder.pkl"

# Recognition parameters
SEQUENCE_LENGTH = 150  # Must match training
PREDICTION_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for prediction
SEQUENCE_BUFFER_SIZE = SEQUENCE_LENGTH
PREDICTION_SMOOTHING_FRAMES = 5  # Average predictions over N frames

# Initialize prediction buffers
sequence_buffer = deque(maxlen=SEQUENCE_BUFFER_SIZE)
prediction_history = deque(maxlen=PREDICTION_SMOOTHING_FRAMES)
current_prediction = "No gesture detected"
prediction_confidence = 0.0

# -------------------------------
# Load Trained Model and Components
# -------------------------------
print("Loading trained model and components...")

try:
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found!")
        print("Please run the training script first to create the model.")
        exit(1)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully: {MODEL_PATH}")
    
    # Load scaler
    if not os.path.exists(SCALER_PATH):
        print(f"ERROR: Scaler file '{SCALER_PATH}' not found!")
        exit(1)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded successfully: {SCALER_PATH}")
    
    # Load label encoder
    if not os.path.exists(LABEL_ENCODER_PATH):
        print(f"ERROR: Label encoder file '{LABEL_ENCODER_PATH}' not found!")
        exit(1)
    
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"Label encoder loaded successfully: {LABEL_ENCODER_PATH}")
    
    print(f"\nModel ready for prediction!")
    print(f"Available gestures: {list(label_encoder.classes_)}")
    print(f"Model expects sequences of {SEQUENCE_LENGTH} frames")
    
except Exception as e:
    print(f"ERROR loading model components: {e}")
    exit(1)

# -------------------------------
# Initialize MediaPipe Hands           
# -------------------------------
print("Initializing MediaPipe Hands...")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Focus on single hand for better performance
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)

# -------------------------------
# Hand size estimation constants (copied from training)
# -------------------------------
AVERAGE_HAND_SPAN = 18.0
AVERAGE_HAND_LENGTH = 18.5
AVERAGE_PALM_WIDTH = 8.5
CAMERA_FOV_DEGREES = 60
FOCAL_LENGTH_PIXELS = 500

# Calibration system
CALIBRATION_FRAMES = 60
calibrated_palm_measurements = None
calibration_data = []
calibration_mode = True
calibration_frame_count = 0

print("CALIBRATION MODE ACTIVE")
print("Please hold your hand naturally in front of the camera")
print("Rotate your hand in different orientations")
print(f"Collecting data for {CALIBRATION_FRAMES} frames...")
print("Press 'r' to restart calibration, 's' to skip calibration")

# -------------------------------
# Feature Extraction Functions (copied from training file)
# -------------------------------
def calculate_all_palm_measurements(landmarks, frame_width, frame_height):
    """Calculate multiple palm measurements"""
    measurements = {}
    positions = {}
    
    wrist = landmarks[0]
    thumb_cmc = landmarks[1]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]
    
    # Convert to pixel coordinates
    points = {}
    for name, landmark in [
        ('wrist', wrist), ('thumb_cmc', thumb_cmc), ('index_mcp', index_mcp),
        ('middle_mcp', middle_mcp), ('ring_mcp', ring_mcp), ('pinky_mcp', pinky_mcp)
    ]:
        points[name] = (landmark.x * frame_width, landmark.y * frame_height)
    
    # Index MCP to Pinky MCP
    index_to_pinky = math.sqrt(
        (points['index_mcp'][0] - points['pinky_mcp'][0])**2 + 
        (points['index_mcp'][1] - points['pinky_mcp'][1])**2
    )
    measurements['index_to_pinky'] = index_to_pinky
    positions['index_to_pinky'] = (points['index_mcp'], points['pinky_mcp'])
    
    # Thumb CMC to Pinky MCP
    thumb_to_pinky = math.sqrt(
        (points['thumb_cmc'][0] - points['pinky_mcp'][0])**2 + 
        (points['thumb_cmc'][1] - points['pinky_mcp'][1])**2
    )
    measurements['thumb_to_pinky'] = thumb_to_pinky
    positions['thumb_to_pinky'] = (points['thumb_cmc'], points['pinky_mcp'])
    
    # Wrist to MCP center
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
    """Select the measurement with highest ratio to calibrated value"""
    if calibrated_measurements is None:
        return 'index_to_pinky', current_measurements['index_to_pinky'], 1.0
    
    best_measurement = None
    best_ratio = None
    highest_ratio = float('-inf')
    
    for measurement_name in current_measurements:
        if measurement_name in calibrated_measurements:
            current_val = current_measurements[measurement_name]
            calibrated_val = calibrated_measurements[measurement_name]
            
            if calibrated_val > 0:
                ratio = current_val / calibrated_val
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_ratio = ratio
                    best_measurement = measurement_name
    
    if best_measurement is None:
        return 'index_to_pinky', current_measurements['index_to_pinky'], 1.0
    
    return best_measurement, current_measurements[best_measurement], best_ratio

def calculate_palm_orientation(landmarks, frame_width, frame_height):
    """Calculate palm orientation angle"""
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    
    wrist_x, wrist_y = wrist.x * frame_width, wrist.y * frame_height
    index_x, index_y = index_mcp.x * frame_width, index_mcp.y * frame_height
    middle_x, middle_y = middle_mcp.x * frame_width, middle_mcp.y * frame_height
    
    palm_center_x = (index_x + middle_x) / 2
    palm_center_y = (index_y + middle_y) / 2
    
    palm_vector_x = palm_center_x - wrist_x
    palm_vector_y = palm_center_y - wrist_y
    
    angle_radians = math.atan2(palm_vector_x, -palm_vector_y)
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def estimate_distance_from_multi_edge_palm(current_measurements, calibrated_measurements, frame_width, assumed_focal_length=500):
    """Estimate distance using the best palm measurement"""
    if calibrated_measurements is None:
        return None, None, None, None
    
    best_measurement_name, best_current_value, ratio = select_best_measurement(
        current_measurements, calibrated_measurements)
    
    if best_current_value < 5:
        return None, None, None, None
    
    reference_distance_cm = 40.0
    calibrated_value = calibrated_measurements[best_measurement_name]
    distance_cm = (calibrated_value * reference_distance_cm) / best_current_value
    
    return distance_cm, best_measurement_name, best_current_value, ratio

def calculate_palm_plane_reference(landmarks, frame_width, frame_height):
    """Calculate reference palm plane using key palm points"""
    palm_landmarks = [0, 1, 5, 9, 13, 17]  # Wrist, Thumb CMC, and all MCP joints
    
    palm_z_sum = 0
    for landmark_idx in palm_landmarks:
        palm_z_sum += landmarks[landmark_idx].z
    
    palm_reference_z = palm_z_sum / len(palm_landmarks)
    return palm_reference_z

def calculate_enhanced_fingertip_depth(landmarks, fingertip_idx, hand_distance_cm, frame_width, frame_height):
    """Calculate fingertip depth using enhanced methods"""
    if hand_distance_cm is None:
        return None
    
    fingertip = landmarks[fingertip_idx]
    palm_reference_z = calculate_palm_plane_reference(landmarks, frame_width, frame_height)
    relative_z = fingertip.z - palm_reference_z
    z_scale_factor = hand_distance_cm * 0.3
    scaled_relative_z = relative_z * z_scale_factor
    final_depth = hand_distance_cm + scaled_relative_z
    
    return final_depth, {
        'palm_reference_z': palm_reference_z,
        'relative_z': relative_z,
        'scaled_relative_z': scaled_relative_z
    }

def extract_features_from_frame(frame_data):
    """Extract numerical features from a single frame (same as training)"""
    features = []

    hands = frame_data.get('hands', [])
    if hands:
        hand_data = hands[0]  # Use first hand

        # Basic hand info (3 values)
        features.extend([
            hand_data.get('hand_confidence', 0),
            hand_data.get('hand_distance_cm', 40),
            hand_data.get('palm_orientation', 0)
        ])

        # Palm measurements (3 values)
        palm_measurements = hand_data.get('palm_measurements', {})
        features.extend([
            palm_measurements.get('index_to_pinky', 80),
            palm_measurements.get('thumb_to_pinky', 120),
            palm_measurements.get('wrist_to_mcp', 90)
        ])

        # Landmarks 3D (63 values)
        landmarks_3d = hand_data.get('landmarks_3d', [0] * 63)
        if len(landmarks_3d) >= 63:
            features.extend(landmarks_3d[:63])
        else:
            features.extend(landmarks_3d + [0] * (63 - len(landmarks_3d)))

        # Finger angles (5 values)
        finger_angles = hand_data.get('finger_angles', [0] * 5)
        if len(finger_angles) >= 5:
            features.extend(finger_angles[:5])
        else:
            features.extend(finger_angles + [0] * (5 - len(finger_angles)))

        # Distances (5 values)
        distances = hand_data.get('distances', [0] * 5)
        if len(distances) >= 5:
            features.extend(distances[:5])
        else:
            features.extend(distances + [0] * (5 - len(distances)))

        # Motion features (4 values)
        motion = hand_data.get('motion', {})
        features.extend([
            motion.get('palm_velocity', 0),
            motion.get('overall_motion_magnitude', 0),
            len(motion.get('fingertip_velocities', {})),
            abs(sum(motion.get('motion_direction', [0, 0, 0])))
        ])

        # Fingertip positions (15 values)
        fingertip_positions = hand_data.get('fingertip_positions', {})
        fingertip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for finger_name in fingertip_names:
            if finger_name in fingertip_positions:
                pos = fingertip_positions[finger_name]
                features.extend([pos.get('x', 0), pos.get('y', 0), pos.get('z', 40)])
            else:
                features.extend([0, 0, 40])
    else:
        # If no hands detected, return zero features (98 values total)
        features = [0] * 98

    return features

def extract_comprehensive_hand_features(hand_landmarks, handedness, hand_distance_cm, frame_width, frame_height, focal_length=500):
    """Extract comprehensive features (same as training)"""
    features = {
        'hand_label': handedness.classification[0].label,
        'hand_confidence': handedness.classification[0].score,
        'hand_distance_cm': hand_distance_cm,
        'palm_measurements': {},
        'landmarks_3d': [],
        'landmarks_normalized': [],
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
    
    # Extract all landmark positions
    for i, landmark in enumerate(landmarks):
        features['landmarks_normalized'].extend([landmark.x, landmark.y, landmark.z])
        
        if hand_distance_cm is not None:
            pixel_x = landmark.x * frame_width
            pixel_y = landmark.y * frame_height
            centered_x = pixel_x - (frame_width / 2)
            centered_y = pixel_y - (frame_height / 2)
            real_x_cm = (centered_x * hand_distance_cm) / focal_length
            real_y_cm = (centered_y * hand_distance_cm) / focal_length
            
            if i in [4, 8, 12, 16, 20]:  # Fingertips
                real_z_cm, _ = calculate_enhanced_fingertip_depth(landmarks, i, hand_distance_cm, frame_width, frame_height)
                if real_z_cm is None:
                    real_z_cm = hand_distance_cm
            else:
                palm_ref_z = calculate_palm_plane_reference(landmarks, frame_width, frame_height)
                relative_z = landmark.z - palm_ref_z
                real_z_cm = hand_distance_cm + (relative_z * hand_distance_cm * 0.2)
            
            features['landmarks_3d'].extend([real_x_cm, real_y_cm, real_z_cm])
        else:
            features['landmarks_3d'].extend([landmark.x, landmark.y, landmark.z])
    
    # Extract fingertip positions
    fingertips = [4, 8, 12, 16, 20]
    fingertip_names = ["thumb", "index", "middle", "ring", "pinky"]
    
    for tip_idx, name in zip(fingertips, fingertip_names):
        if hand_distance_cm is not None:
            pixel_x = landmarks[tip_idx].x * frame_width
            pixel_y = landmarks[tip_idx].y * frame_height
            centered_x = pixel_x - (frame_width / 2)
            centered_y = pixel_y - (frame_height / 2)
            real_x_cm = (centered_x * hand_distance_cm) / focal_length
            real_y_cm = (centered_y * hand_distance_cm) / focal_length
            real_z_cm, _ = calculate_enhanced_fingertip_depth(landmarks, tip_idx, hand_distance_cm, frame_width, frame_height)
            if real_z_cm is None:
                real_z_cm = hand_distance_cm
                
            features['fingertip_positions'][name] = {
                'x': real_x_cm, 'y': real_y_cm, 'z': real_z_cm
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
    
    # Calculate key distances
    palm_center = landmarks[0]  # Wrist as palm reference
    fingertips = [4, 8, 12, 16, 20]
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
    
    # Calculate palm motion
    if 'landmarks_3d' in current_features and 'landmarks_3d' in previous_features:
        if len(current_features['landmarks_3d']) >= 3 and len(previous_features['landmarks_3d']) >= 3:
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

# -------------------------------
# Prediction Functions
# -------------------------------
def preprocess_sequence(sequence_data, max_length=SEQUENCE_LENGTH):
    """Preprocess sequence for model prediction (same as training)"""
    processed_sequences = []
    
    for i, sequence in enumerate([sequence_data]):
        sequence = np.array(sequence, dtype=np.float32)
        
        # Truncate if too long
        if len(sequence) > max_length:
            start_idx = (len(sequence) - max_length) // 2
            sequence = sequence[start_idx:start_idx + max_length]
        
        if len(sequence) > 0:
            base_frame = sequence[0].copy()
            
            # Apply translation relative to first frame
            landmark_indices = np.arange(6, 69)     # landmarks 3D
            fingertip_indices = np.arange(83, 98)   # fingertips 3D
            
            for idx_group in [landmark_indices, fingertip_indices]:
                for j in range(0, len(idx_group), 3):
                    if j+2 < len(idx_group):
                        sequence[:, idx_group[j]]   -= base_frame[idx_group[j]]
                        sequence[:, idx_group[j+1]] -= base_frame[idx_group[j+1]]
                        sequence[:, idx_group[j+2]] -= base_frame[idx_group[j+2]]
            
            # Normalize hand distance
            sequence[:, 1] -= base_frame[1]
        
        processed_sequences.append(sequence)
    
    # Pad sequences
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        processed_sequences,
        maxlen=max_length,
        dtype='float32',
        padding='post'
    )
    
    return X_padded

def predict_gesture(sequence_buffer):
    """Make prediction from current sequence buffer"""
    if len(sequence_buffer) < 10:  # Need minimum frames
        return "Collecting frames...", 0.0
    
    try:
        # Convert buffer to sequence
        sequence_data = []
        for frame_features in sequence_buffer:
            sequence_data.append(frame_features)
        
        # Preprocess sequence
        X_processed = preprocess_sequence(sequence_data)
        
        # Normalize features using saved scaler
        n_samples, n_timesteps, n_features = X_processed.shape
        X_reshaped = X_processed.reshape(-1, n_features)
        X_normalized = scaler.transform(X_reshaped)
        X_final = X_normalized.reshape(n_samples, n_timesteps, n_features)
        
        # Make prediction
        predictions = model.predict(X_final, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        if confidence > PREDICTION_CONFIDENCE_THRESHOLD:
            gesture_name = label_encoder.inverse_transform([predicted_class_idx])[0]
            return gesture_name, confidence
        else:
            return "Low confidence", confidence
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction error", 0.0

def smooth_predictions(new_prediction, new_confidence):
    """Smooth predictions over multiple frames"""
    global prediction_history, current_prediction, prediction_confidence
    
    prediction_history.append((new_prediction, new_confidence))
    
    # Find most common prediction with sufficient confidence
    if len(prediction_history) >= PREDICTION_SMOOTHING_FRAMES:
        recent_predictions = list(prediction_history)
        
        # Count predictions with good confidence
        prediction_counts = {}
        total_confidence = {}
        
        for pred, conf in recent_predictions:
            if conf > PREDICTION_CONFIDENCE_THRESHOLD:
                if pred not in prediction_counts:
                    prediction_counts[pred] = 0
                    total_confidence[pred] = 0
                prediction_counts[pred] += 1
                total_confidence[pred] += conf
        
        # Find most frequent high-confidence prediction
        if prediction_counts:
            best_prediction = max(prediction_counts.keys(), key=lambda x: prediction_counts[x])
            avg_confidence = total_confidence[best_prediction] / prediction_counts[best_prediction]
            
            # Only update if we have enough consistent predictions
            if prediction_counts[best_prediction] >= PREDICTION_SMOOTHING_FRAMES // 2:
                current_prediction = best_prediction
                prediction_confidence = avg_confidence
        else:
            current_prediction = "No confident prediction"
            prediction_confidence = 0.0
    
    return current_prediction, prediction_confidence

# -------------------------------
# Open webcam
# -------------------------------
print("Opening webcam...")
cap = cv2.VideoCapture(2    )

if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    exit(1)

ret, test_frame = cap.read()
if not ret or test_frame is None:
    print("ERROR: Could not read from webcam!")
    cap.release()
    exit(1)

frame_height, frame_width = test_frame.shape[:2]
print(f"Webcam working! Frame size: {frame_width}x{frame_height}")

# FPS tracking
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0
frame_count = 0

# Motion tracking
previous_hand_features = None
last_frame_time = time.time()

print("Starting live gesture recognition...")
print("Controls: Press 'q' or ESC to quit, 'r' to restart calibration, 's' to skip calibration")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
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

        try:
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            hands_detected = 0
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hands_detected = len(results.multi_hand_landmarks)
                
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw hand connections
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Calculate palm measurements
                    current_measurements, measurement_positions = calculate_all_palm_measurements(
                        hand_landmarks.landmark, frame_width, frame_height)
                    
                    # Calibration mode
                    if calibration_mode:
                        calibration_frame_count += 1
                        calibration_data.append(current_measurements.copy())
                        
                        # Draw calibration progress
                        progress = calibration_frame_count / CALIBRATION_FRAMES
                        cv2.rectangle(frame, (50, 50), (50 + int(300 * progress), 80), (0, 255, 0), -1)
                        cv2.rectangle(frame, (50, 50), (350, 80), (255, 255, 255), 2)
                        
                        calibration_text = f"CALIBRATING... {calibration_frame_count}/{CALIBRATION_FRAMES}"
                        cv2.putText(frame, calibration_text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        if calibration_frame_count >= CALIBRATION_FRAMES:
                            # Complete calibration
                            calibrated_palm_measurements = {}
                            for measurement_name in current_measurements.keys():
                                values = [frame_data[measurement_name] for frame_data in calibration_data]
                                calibrated_palm_measurements[measurement_name] = sum(values) / len(values)
                            
                            calibration_mode = False
                            print(f"\nCALIBRATION COMPLETE!")
                            print("*** LIVE GESTURE RECOGNITION ACTIVE ***")
                    
                    # Normal operation mode
                    else:
                        # Calculate distance
                        hand_distance, best_measurement, best_value, ratio = estimate_distance_from_multi_edge_palm(
                            current_measurements, calibrated_palm_measurements, frame_width, FOCAL_LENGTH_PIXELS
                        )
                        
                        if hand_distance is not None:
                            # Extract comprehensive features
                            hand_features = extract_comprehensive_hand_features(
                                hand_landmarks, handedness, hand_distance, frame_width, frame_height, FOCAL_LENGTH_PIXELS)
                            
                            # Calculate motion features
                            motion_features = calculate_motion_features(hand_features, previous_hand_features, dt)
                            hand_features['motion'] = motion_features
                            hand_features['timestamp'] = current_time
                            hand_features['dt'] = dt
                            
                            # Extract frame features for model
                            frame_data = {'hands': [hand_features]}
                            frame_features = extract_features_from_frame(frame_data)
                            
                            # Add to sequence buffer
                            sequence_buffer.append(frame_features)
                            
                            # Make prediction if we have enough frames
                            raw_prediction, raw_confidence = predict_gesture(sequence_buffer)
                            smoothed_prediction, smoothed_confidence = smooth_predictions(raw_prediction, raw_confidence)
                            
                            # Update previous features
                            previous_hand_features = hand_features
                            
                            # Draw visualization
                            # Draw best measurement line (green)
                            if best_measurement in measurement_positions:
                                start_pos, end_pos = measurement_positions[best_measurement]
                                cv2.line(frame, (int(start_pos[0]), int(start_pos[1])), 
                                        (int(end_pos[0]), int(end_pos[1])), (0, 255, 0), 3)
                            
                            # Draw other measurements (dimmed)
                            colors = [(100, 100, 255), (100, 255, 255), (255, 100, 255)]
                            for i, (measurement_name, positions) in enumerate(measurement_positions.items()):
                                if measurement_name != best_measurement:
                                    color = colors[i % len(colors)]
                                    start_pos, end_pos = positions
                                    cv2.line(frame, (int(start_pos[0]), int(start_pos[1])), 
                                            (int(end_pos[0]), int(end_pos[1])), color, 1)
                            
                            # Display hand info
                            hand_label = handedness.classification[0].label
                            distance_text = f"{hand_label}: {hand_distance:.1f}cm"
                            cv2.putText(frame, distance_text, (10, 120), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                            
                            # Display motion info
                            motion_text = f"Motion: {motion_features['overall_motion_magnitude']:.1f} cm/s"
                            cv2.putText(frame, motion_text, (10, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            
                            # Display buffer status
                            buffer_text = f"Buffer: {len(sequence_buffer)}/{SEQUENCE_BUFFER_SIZE}"
                            cv2.putText(frame, buffer_text, (10, 180), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Clear sequence buffer if no hands detected
            else:
                if len(sequence_buffer) > 0:
                    sequence_buffer.clear()
                    current_prediction = "No hand detected"
                    prediction_confidence = 0.0

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")

        # -------------------------------
        # Draw UI Elements
        # -------------------------------
        
        # FPS
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hands detected
        cv2.putText(frame, f"Hands: {hands_detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if calibration_mode:
            cv2.putText(frame, "Live Gesture Recognition - Calibrating", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Live Gesture Recognition - Active", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # -------------------------------
            # MAIN PREDICTION DISPLAY
            # -------------------------------
            
            # Background box for prediction
            prediction_box_height = 120
            prediction_box_width = frame_width - 20
            box_y = frame_height - prediction_box_height - 10
            
            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, box_y), (prediction_box_width, frame_height - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Main prediction text
            if prediction_confidence > PREDICTION_CONFIDENCE_THRESHOLD:
                prediction_display = current_prediction.replace('_', ' ').title()
                prediction_color = (0, 255, 0)  # Green for confident prediction
            else:
                prediction_display = current_prediction
                prediction_color = (0, 255, 255)  # Yellow for uncertain
            
            # Large prediction text
            font_scale = 1.2
            prediction_text = f"Gesture: {prediction_display}"
            text_size = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = (frame_width - text_size[0]) // 2
            text_y = box_y + 40
            
            cv2.putText(frame, prediction_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, prediction_color, 2)
            
            # Confidence bar
            confidence_text = f"Confidence: {prediction_confidence:.1%}"
            cv2.putText(frame, confidence_text, (20, text_y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Confidence bar visualization
            bar_width = 300
            bar_height = 15
            bar_x = 20
            bar_y = text_y + 45
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Confidence fill
            fill_width = int(bar_width * prediction_confidence)
            bar_color = (0, 255, 0) if prediction_confidence > PREDICTION_CONFIDENCE_THRESHOLD else (0, 255, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)
            
            # Bar border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
            
            # Threshold line
            threshold_x = bar_x + int(bar_width * PREDICTION_CONFIDENCE_THRESHOLD)
            cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), (255, 255, 255), 2)
            
            # Available gestures (small text on side)
            gesture_list_x = frame_width - 250
            gesture_list_y = 120
            cv2.putText(frame, "Trained Gestures:", (gesture_list_x, gesture_list_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            for i, gesture in enumerate(label_encoder.classes_[:10]):  # Show first 10 gestures
                display_name = gesture.replace('_', ' ').title()
                color = (0, 255, 0) if gesture == current_prediction else (150, 150, 150)
                cv2.putText(frame, display_name, (gesture_list_x, gesture_list_y + 20 + i*15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
            # Instructions
            instructions_y = 220
            cv2.putText(frame, "Hold gesture steady for recognition", (10, instructions_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Move hand naturally for motion gestures", (10, instructions_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display frame
        cv2.imshow("Live Gesture Recognition", frame)

        # -------------------------------
        # Handle keyboard input
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
            sequence_buffer.clear()
            prediction_history.clear()
            current_prediction = "Calibrating..."
            prediction_confidence = 0.0
        elif key == ord('s') and calibration_mode:  # 's' to skip calibration
            print("\nSkipping calibration - using default values")
            calibrated_palm_measurements = {
                'index_to_pinky': 80,
                'thumb_to_pinky': 120,
                'wrist_to_mcp': 90
            }
            calibration_mode = False
            print("*** LIVE GESTURE RECOGNITION NOW ACTIVE ***")

    print("Exited main loop normally")

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Live gesture recognition ended.")

print("Program completed.")
input("Press Enter to close console...")