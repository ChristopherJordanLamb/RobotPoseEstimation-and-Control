import cv2
import numpy as np
import tensorflow as tf
import pickle
import mediapipe as mp
import time
import math
from collections import deque
import os

print("Real-time Gesture Recognition System")
print("="*50)

# Load trained model and preprocessing components
MODEL_PATH = "gesture_model.h5"
SCALER_PATH = "gesture_scaler.pkl"
LABEL_ENCODER_PATH = "gesture_label_encoder.pkl"

# Configuration
SEQUENCE_LENGTH = 150  # Should match training
CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5  # Number of predictions to average

class GestureRecognizer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Single hand for simplicity
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # Palm calibration (simplified - you might want to load from your calibration)
        self.calibrated_palm_measurements = {
            'index_to_pinky': 80,
            'thumb_to_pinky': 120,
            'wrist_to_mcp': 90
        }
        
        self.load_model_components()
    
    def load_model_components(self):
        """Load the trained model and preprocessing components"""
        try:
            print("Loading trained model...")
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully!")
            
            print("Loading scaler...")
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("Loading label encoder...")
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print(f"Classes: {list(self.label_encoder.classes_)}")
            print("All components loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model components: {e}")
            print("Make sure you've trained the model first!")
            return False
        return True
    
    def calculate_all_palm_measurements(self, landmarks, frame_width, frame_height):
        """Calculate palm measurements (simplified version)"""
        measurements = {}
        
        # Get key landmark positions
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
        
        # Calculate measurements
        measurements['index_to_pinky'] = math.sqrt(
            (points['index_mcp'][0] - points['pinky_mcp'][0])**2 + 
            (points['index_mcp'][1] - points['pinky_mcp'][1])**2
        )
        
        measurements['thumb_to_pinky'] = math.sqrt(
            (points['thumb_cmc'][0] - points['pinky_mcp'][0])**2 + 
            (points['thumb_cmc'][1] - points['pinky_mcp'][1])**2
        )
        
        # Palm length
        mcp_center_x = (points['index_mcp'][0] + points['middle_mcp'][0] + 
                        points['ring_mcp'][0] + points['pinky_mcp'][0]) / 4
        mcp_center_y = (points['index_mcp'][1] + points['middle_mcp'][1] + 
                        points['ring_mcp'][1] + points['pinky_mcp'][1]) / 4
        
        measurements['wrist_to_mcp'] = math.sqrt(
            (points['wrist'][0] - mcp_center_x)**2 + 
            (points['wrist'][1] - mcp_center_y)**2
        )
        
        return measurements
    
    def calculate_palm_orientation(self, landmarks, frame_width, frame_height):
        """Calculate palm orientation"""
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
    
    def estimate_hand_distance(self, current_measurements):
        """Estimate hand distance using palm measurements"""
        best_measurement_name = 'index_to_pinky'  # Simplified
        best_current_value = current_measurements.get(best_measurement_name, 80)
        
        if best_current_value < 5:
            return 40.0  # Default distance
        
        reference_distance_cm = 40.0
        calibrated_value = self.calibrated_palm_measurements[best_measurement_name]
        distance_cm = (calibrated_value * reference_distance_cm) / best_current_value
        
        return distance_cm
    
    def extract_features_from_hand(self, hand_landmarks, handedness, frame_width, frame_height):
        """Extract features from detected hand (matching training data format)"""
        landmarks = hand_landmarks.landmark
        
        # Calculate measurements
        palm_measurements = self.calculate_all_palm_measurements(landmarks, frame_width, frame_height)
        palm_orientation = self.calculate_palm_orientation(landmarks, frame_width, frame_height)
        hand_distance = self.estimate_hand_distance(palm_measurements)
        
        features = []
        
        # Basic hand info
        features.extend([
            handedness.classification[0].score,  # hand_confidence
            hand_distance,  # hand_distance_cm
            palm_orientation  # palm_orientation
        ])
        
        # Palm measurements (3 values)
        features.extend([
            palm_measurements.get('index_to_pinky', 80),
            palm_measurements.get('thumb_to_pinky', 120),
            palm_measurements.get('wrist_to_mcp', 90)
        ])
        
        # Landmarks 3D (21 landmarks * 3 coordinates = 63 values)
        landmarks_3d = []
        focal_length = 500
        for landmark in landmarks:
            pixel_x = landmark.x * frame_width
            pixel_y = landmark.y * frame_height
            centered_x = pixel_x - (frame_width / 2)
            centered_y = pixel_y - (frame_height / 2)
            real_x_cm = (centered_x * hand_distance) / focal_length
            real_y_cm = (centered_y * hand_distance) / focal_length
            real_z_cm = hand_distance + (landmark.z * hand_distance * 0.2)
            landmarks_3d.extend([real_x_cm, real_y_cm, real_z_cm])
        
        features.extend(landmarks_3d[:63])
        
        # Finger angles (5 values)
        finger_joints = [
            [1, 2, 3, 4],      # Thumb
            [5, 6, 7, 8],      # Index
            [9, 10, 11, 12],   # Middle
            [13, 14, 15, 16],  # Ring
            [17, 18, 19, 20]   # Pinky
        ]
        
        finger_angles = []
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
                    finger_angles.append(angle)
                else:
                    finger_angles.append(0)
            else:
                finger_angles.append(0)
        
        features.extend(finger_angles[:5])
        
        # Distances (5 values)
        palm_center = landmarks[0]  # Wrist
        fingertips = [4, 8, 12, 16, 20]
        distances = []
        for tip_idx in fingertips:
            tip = landmarks[tip_idx]
            distance = math.sqrt(
                (tip.x - palm_center.x)**2 + 
                (tip.y - palm_center.y)**2 + 
                (tip.z - palm_center.z)**2
            )
            distances.append(distance)
        
        features.extend(distances[:5])
        
        # Motion features (4 values) - simplified for real-time
        # In real-time, we'll compute these from frame-to-frame differences
        features.extend([0, 0, 0, 0])  # Will be updated with actual motion later
        
        # Fingertip positions (5 fingertips * 3 coordinates = 15 values)
        fingertip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        fingertip_indices = [4, 8, 12, 16, 20]
        
        for tip_idx in fingertip_indices:
            landmark = landmarks[tip_idx]
            pixel_x = landmark.x * frame_width
            pixel_y = landmark.y * frame_height
            centered_x = pixel_x - (frame_width / 2)
            centered_y = pixel_y - (frame_height / 2)
            real_x_cm = (centered_x * hand_distance) / focal_length
            real_y_cm = (centered_y * hand_distance) / focal_length
            real_z_cm = hand_distance + (landmark.z * hand_distance * 0.2)
            features.extend([real_x_cm, real_y_cm, real_z_cm])
        
        return features
    
    def update_motion_features(self):
        """Update motion features based on recent frames"""
        if len(self.feature_buffer) < 2:
            return
        
        current_frame = self.feature_buffer[-1]
        previous_frame = self.feature_buffer[-2]
        
        # Calculate simple motion metrics
        # Palm position (using first 3D landmark - wrist)
        current_palm = current_frame[6:9]  # landmarks_3d[0:3]
        previous_palm = previous_frame[6:9]
        
        palm_velocity = math.sqrt(sum((c - p)**2 for c, p in zip(current_palm, previous_palm)))
        
        # Update motion features in current frame
        motion_start_idx = 81  # Index where motion features start
        current_frame[motion_start_idx] = palm_velocity
        current_frame[motion_start_idx + 1] = palm_velocity  # overall_motion_magnitude
        current_frame[motion_start_idx + 2] = 1 if palm_velocity > 0.5 else 0  # moving fingertips count
        current_frame[motion_start_idx + 3] = palm_velocity  # total motion vector magnitude
    
    def predict_gesture(self):
        """Predict gesture from current feature buffer"""
        if len(self.feature_buffer) < SEQUENCE_LENGTH