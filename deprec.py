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
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Starting Enhanced Live Gesture Recognition...")

# -------------------------------
# Enhanced Model Loading Configuration
# -------------------------------
MODEL_DIR = "models/20250818_234215"
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.yaml")

# Fallback paths for backward compatibility
FALLBACK_SCALER = "gesture_scaler.pkl"
FALLBACK_LABEL_ENCODER = "gesture_label_encoder.pkl"

# Recognition parameters - these will be updated from config if available
SEQUENCE_LENGTH = 150
PREDICTION_CONFIDENCE_THRESHOLD = 0.6
SEQUENCE_BUFFER_SIZE = SEQUENCE_LENGTH
PREDICTION_SMOOTHING_FRAMES = 7  # Increased for better smoothing
PREDICTION_CONSENSUS_THRESHOLD = 0.4  # Require 40% consensus for prediction

# Enhanced prediction buffers
sequence_buffer = deque(maxlen=SEQUENCE_BUFFER_SIZE)
prediction_history = deque(maxlen=PREDICTION_SMOOTHING_FRAMES)
confidence_history = deque(maxlen=PREDICTION_SMOOTHING_FRAMES)
current_prediction = "No gesture detected"
prediction_confidence = 0.0
stable_prediction_count = 0
last_stable_prediction = ""

# Performance tracking
frame_processing_times = deque(maxlen=30)
prediction_times = deque(maxlen=30)

class ModelConfig:
    """Configuration class to match training setup"""
    def __init__(self):
        self.max_sequence_length = 150
        self.feature_dim = 98
        self.prediction_confidence_threshold = 0.6
        
    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML if available"""
        try:
            import yaml
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                self.max_sequence_length = config_dict.get('max_sequence_length', 150)
                self.feature_dim = config_dict.get('feature_dim', 98)
                logger.info(f"Loaded config: seq_len={self.max_sequence_length}, features={self.feature_dim}")
        except ImportError:
            logger.warning("PyYAML not available, using default config")
        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")
class EnhancedPredictor:
    """Enhanced prediction system with better smoothing and confidence"""
    
    def __init__(self, model, scaler, label_encoder, config: ModelConfig):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.config = config
        
        # Prediction buffers
        self.sequence_buffer = deque(maxlen=config.max_sequence_length)
        self.prediction_history = deque(maxlen=PREDICTION_SMOOTHING_FRAMES)
        self.confidence_history = deque(maxlen=PREDICTION_SMOOTHING_FRAMES)
        
        # State tracking
        self.current_prediction = "No gesture detected"
        self.prediction_confidence = 0.0
        self.stable_prediction_count = 0
        self.last_stable_prediction = ""
        
        # Performance tracking
        self.prediction_times = deque(maxlen=30)
    
    def predict_gesture(self, frame_features: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Make prediction with enhanced processing"""
        prediction_start_time = time.time()
        
        # Add frame to sequence buffer
        self.sequence_buffer.append(frame_features)
        
        # Need minimum frames for prediction
        if len(self.sequence_buffer) < 10:
            return "Collecting frames...", 0.0, {'raw_predictions': [], 'processing_time': 0.0}
        
        try:
            # Preprocess sequence
            sequence_data = list(self.sequence_buffer)
            X_processed = self._preprocess_sequence(sequence_data)
            
            # Normalize features using saved scaler
            n_samples, n_timesteps, n_features = X_processed.shape
            X_reshaped = X_processed.reshape(-1, n_features)
            
            # Handle potential scaling issues
            try:
                X_normalized = self.scaler.transform(X_reshaped)
            except Exception as e:
                logger.warning(f"Scaler transform error: {e}")
                # Fallback: use robust normalization
                X_normalized = self._robust_normalize(X_reshaped)
            
            X_final = X_normalized.reshape(n_samples, n_timesteps, n_features)
            
            # Make prediction
            predictions = self.model.predict(X_final, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get gesture name
            gesture_name = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            processing_time = time.time() - prediction_start_time
            self.prediction_times.append(processing_time)
            
            # Apply smoothing
            smoothed_prediction, smoothed_confidence = self._smooth_predictions(gesture_name, confidence)
            
            prediction_info = {
                'raw_predictions': predictions[0].tolist(),
                'top_3_predictions': self._get_top_k_predictions(predictions[0], k=3),
                'processing_time': processing_time,
                'sequence_length': len(self.sequence_buffer),
                'avg_processing_time': np.mean(list(self.prediction_times)) if self.prediction_times else 0.0
            }
            
            return smoothed_prediction, smoothed_confidence, prediction_info
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Prediction error", 0.0, {'error': str(e), 'processing_time': 0.0}
    
    def _preprocess_sequence(self, sequence_data: List[np.ndarray]) -> np.ndarray:
        """Preprocess sequence for prediction"""
        if not sequence_data:
            return np.zeros((1, self.config.max_sequence_length, self.config.feature_dim))
        
        # Convert to numpy array and pad
        sequence = np.array(sequence_data, dtype=np.float32)
        
        if len(sequence) > self.config.max_sequence_length:
            start_idx = (len(sequence) - self.config.max_sequence_length) // 2
            sequence = sequence[start_idx:start_idx + self.config.max_sequence_length]
        
        # Pad to required length
        if len(sequence) < self.config.max_sequence_length:
            padding = np.zeros((self.config.max_sequence_length - len(sequence), self.config.feature_dim))
            sequence = np.vstack([sequence, padding])
        
        return sequence.reshape(1, self.config.max_sequence_length, self.config.feature_dim)
    
    def _robust_normalize(self, X: np.ndarray) -> np.ndarray:
        """Fallback robust normalization"""
        try:
            # Use median and MAD for robust scaling
            median = np.median(X, axis=0)
            mad = np.median(np.abs(X - median), axis=0)
            # Avoid division by zero
            mad[mad == 0] = 1.0
            return (X - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
        except:
            # Ultimate fallback: just return original data
            return X
    
    def _get_top_k_predictions(self, predictions: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Get top K predictions with confidence scores"""
        top_indices = np.argsort(predictions)[-k:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            gesture_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(predictions[idx])
            top_predictions.append({
                'gesture': gesture_name,
                'confidence': confidence
            })
        
        return top_predictions
    
    def _smooth_predictions(self, new_prediction: str, new_confidence: float) -> Tuple[str, float]:
        """Enhanced prediction smoothing with stability tracking"""
        self.prediction_history.append(new_prediction)
        self.confidence_history.append(new_confidence)
        
        # Only consider recent predictions
        if len(self.prediction_history) >= PREDICTION_SMOOTHING_FRAMES:
            recent_predictions = list(self.prediction_history)
            recent_confidences = list(self.confidence_history)
            
            # Count predictions with sufficient confidence
            high_confidence_predictions = {}
            total_confidences = {}
            
            for pred, conf in zip(recent_predictions, recent_confidences):
                if conf > PREDICTION_CONFIDENCE_THRESHOLD:
                    if pred not in high_confidence_predictions:
                        high_confidence_predictions[pred] = 0
                        total_confidences[pred] = 0.0
                    high_confidence_predictions[pred] += 1
                    total_confidences[pred] += conf
            
            # Find most consistent high-confidence prediction
            if high_confidence_predictions:
                # Get prediction with most occurrences
                most_frequent = max(high_confidence_predictions.keys(), 
                                   key=lambda x: high_confidence_predictions[x])
                
                # Check if it meets consensus threshold
                consensus_ratio = high_confidence_predictions[most_frequent] / len(recent_predictions)
                
                if consensus_ratio >= PREDICTION_CONSENSUS_THRESHOLD:
                    avg_confidence = total_confidences[most_frequent] / high_confidence_predictions[most_frequent]
                    
                    # Update stable prediction tracking
                    if most_frequent == self.last_stable_prediction:
                        self.stable_prediction_count += 1
                    else:
                        self.stable_prediction_count = 1
                        self.last_stable_prediction = most_frequent
                    
                    self.current_prediction = most_frequent
                    self.prediction_confidence = avg_confidence
                    
                    return self.current_prediction, self.prediction_confidence
        
        # Fallback to current prediction
        if new_confidence > PREDICTION_CONFIDENCE_THRESHOLD:
            return new_prediction, new_confidence
        else:
            return "Low confidence", new_confidence
    
    def clear_buffers(self):
        """Clear all prediction buffers"""
        self.sequence_buffer.clear()
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.current_prediction = "No hand detected"
        self.prediction_confidence = 0.0
        self.stable_prediction_count = 0
    

# Initialize config
config = ModelConfig()
config.load_from_yaml(CONFIG_PATH)

# Update global parameters from config
SEQUENCE_LENGTH = config.max_sequence_length
SEQUENCE_BUFFER_SIZE = SEQUENCE_LENGTH

# -------------------------------
# Enhanced Model Loading
# -------------------------------
def load_model_components():
    """Enhanced model loading with better error handling"""
    logger.info("Loading trained model and components...")
    
    components = {}
    
    # Load model
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_path}' not found!")
        logger.info("Please ensure the model directory exists and contains trained model files.")
        return None
    
    try:
        components['model'] = tf.keras.models.load_model(model_path)
        logger.info(f"✓ Model loaded: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Load scaler
    scaler_path = SCALER_PATH if os.path.exists(SCALER_PATH) else FALLBACK_SCALER
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found at '{scaler_path}' or '{FALLBACK_SCALER}'")
        return None
    
    try:
        with open(scaler_path, 'rb') as f:
            components['scaler'] = pickle.load(f)
        logger.info(f"✓ Scaler loaded: {scaler_path}")
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        return None
    
    # Load label encoder
    encoder_path = LABEL_ENCODER_PATH if os.path.exists(LABEL_ENCODER_PATH) else FALLBACK_LABEL_ENCODER
    if not os.path.exists(encoder_path):
        logger.error(f"Label encoder not found at '{encoder_path}' or '{FALLBACK_LABEL_ENCODER}'")
        return None
    
    try:
        with open(encoder_path, 'rb') as f:
            components['label_encoder'] = pickle.load(f)
        logger.info(f"✓ Label encoder loaded: {encoder_path}")
    except Exception as e:
        logger.error(f"Failed to load label encoder: {e}")
        return None
    
    # Display model info
    model = components['model']
    label_encoder = components['label_encoder']
    
    logger.info(f"✓ Model ready - Input shape: {model.input_shape}")
    logger.info(f"✓ Available gestures ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
    logger.info(f"✓ Model parameters: {model.count_params():,}")
    
    return components

# Load model components
model_components = load_model_components()
if model_components is None:
    logger.error("Failed to load model components. Exiting.")
    exit(1)

model = model_components['model']
scaler = model_components['scaler']
label_encoder = model_components['label_encoder']

# -------------------------------
# Enhanced MediaPipe Setup
# -------------------------------
logger.info("Initializing MediaPipe Hands...")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,  # Increased for better detection
    min_tracking_confidence=0.8,   # Increased for better tracking
    model_complexity=1
)

# -------------------------------
# Enhanced Feature Extraction (matching training exactly)
# -------------------------------
class FeatureExtractor:
    """Enhanced feature extractor that exactly matches training pipeline"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Prediction buffers
        self.sequence_buffer = deque(maxlen=config.max_sequence_length)
        self.prediction_history = deque(maxlen=PREDICTION_SMOOTHING_FRAMES)
        self.confidence_history = deque(maxlen=PREDICTION_SMOOTHING_FRAMES)
        
        # State tracking
        self.current_prediction = "No gesture detected"
        self.prediction_confidence = 0.0
        self.stable_prediction_count = 0
        self.last_stable_prediction = ""
        
        # Performance tracking
        self.prediction_times = deque(maxlen=30)
        
        # Add missing attributes that are referenced in the code
        self.feature_names = self._get_feature_names()
        
        # Hand size estimation constants
        self.AVERAGE_HAND_SPAN = 18.0
        self.AVERAGE_HAND_LENGTH = 18.5
        self.AVERAGE_PALM_WIDTH = 8.5
        self.CAMERA_FOV_DEGREES = 60
        self.FOCAL_LENGTH_PIXELS = 500
        
        # Calibration system
        self.CALIBRATION_FRAMES = 60
        self.calibrated_palm_measurements = None
        self.calibration_data = []
        self.calibration_mode = True
        self.calibration_frame_count = 0
        
        logger.info("Feature extractor initialized - calibration mode active")

    def _get_feature_names(self) -> List[str]:
        """Get feature names matching training exactly"""
        names = []
        # Basic hand info (3)
        names.extend(['hand_confidence', 'hand_distance_cm', 'palm_orientation'])
        # Palm measurements (3)
        names.extend(['index_to_pinky', 'thumb_to_pinky', 'wrist_to_mcp'])
        # Landmarks 3D (63)
        for i in range(21):
            names.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
        # Finger angles (5)
        names.extend([f'{finger}_angle' for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']])
        # Distances (5)
        names.extend([f'{finger}_distance' for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']])
        # Motion features (4)
        names.extend(['palm_velocity', 'overall_motion_magnitude', 'moving_fingertips', 'motion_vector_magnitude'])
        # Fingertip positions (15)
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            names.extend([f'{finger}_tip_x', f'{finger}_tip_y', f'{finger}_tip_z'])
        return names
        
    def preprocess_sequence(self, sequence_data: List[np.ndarray]) -> np.ndarray:
        """Preprocess sequence exactly matching training pipeline"""
        if not sequence_data:
            return np.zeros((1, self.config.max_sequence_length, self.config.feature_dim))
        
        processed_sequences = []
        
        # Convert to numpy array
        sequence = np.array(sequence_data, dtype=np.float32)
        
        # Truncate if too long (keep middle portion)
        if len(sequence) > self.config.max_sequence_length:
            start_idx = (len(sequence) - self.config.max_sequence_length) // 2
            sequence = sequence[start_idx:start_idx + self.config.max_sequence_length]
        
        # Apply relative positioning if sequence has frames
        if len(sequence) > 0:
            sequence = self._apply_relative_positioning(sequence)
        
        processed_sequences.append(sequence)
        
        # Pad sequences to uniform length
        X_padded = tf.keras.preprocessing.sequence.pad_sequences(
            processed_sequences,
            maxlen=self.config.max_sequence_length,
            dtype='float32',
            padding='post',
            value=0.0
        )
        
        return X_padded
    
    def _apply_relative_positioning(self, sequence: np.ndarray) -> np.ndarray:
        """Apply relative positioning based on first frame (matching training)"""
        if len(sequence) == 0:
            return sequence
        
        base_frame = sequence[0].copy()
        
        # Landmark indices (3D coordinates) - features 6-68 (63 total)
        landmark_indices = list(range(6, 69))  # 21 landmarks * 3 coords
        fingertip_indices = list(range(83, 98))  # 5 fingertips * 3 coords
        
        # Apply translation relative to first frame for 3D coordinates
        for idx_group in [landmark_indices, fingertip_indices]:
            for j in range(0, len(idx_group), 3):  # Process x, y, z coordinates
                if j + 2 < len(idx_group):
                    sequence[:, idx_group[j]]     -= base_frame[idx_group[j]]      # x
                    sequence[:, idx_group[j + 1]] -= base_frame[idx_group[j + 1]]  # y  
                    sequence[:, idx_group[j + 2]] -= base_frame[idx_group[j + 2]]  # z

        # Normalize hand distance relative to first frame (feature index 1)
        sequence[:, 1] -= base_frame[1]  # hand_distance_cm
        
        return sequence

    def predict_gesture(self, frame_features: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Make prediction with enhanced processing"""
        prediction_start_time = time.time()
        
        # Add frame to sequence buffer
        self.sequence_buffer.append(frame_features)
        
        # Need minimum frames for prediction
        if len(self.sequence_buffer) < 10:
            return "Collecting frames...", 0.0, {'raw_predictions': [], 'processing_time': 0.0}
        
        try:
            # Preprocess sequence
            sequence_data = list(self.sequence_buffer)
            X_processed = self.preprocess_sequence(sequence_data)
            
            # Normalize features using saved scaler
            n_samples, n_timesteps, n_features = X_processed.shape
            X_reshaped = X_processed.reshape(-1, n_features)
            
            # Handle potential scaling issues
            try:
                X_normalized = self.scaler.transform(X_reshaped)
            except Exception as e:
                logger.warning(f"Scaler transform error: {e}")
                # Fallback: use robust normalization
                X_normalized = self._robust_normalize(X_reshaped)
            
            X_final = X_normalized.reshape(n_samples, n_timesteps, n_features)
            
            # Make prediction
            predictions = self.model.predict(X_final, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get gesture name
            gesture_name = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            processing_time = time.time() - prediction_start_time
            self.prediction_times.append(processing_time)
            
            # Apply smoothing
            smoothed_prediction, smoothed_confidence = self._smooth_predictions(gesture_name, confidence)
            
            prediction_info = {
                'raw_predictions': predictions[0].tolist(),
                'top_3_predictions': self._get_top_k_predictions(predictions[0], k=3),
                'processing_time': processing_time,
                'sequence_length': len(self.sequence_buffer),
                'avg_processing_time': np.mean(list(self.prediction_times)) if self.prediction_times else 0.0
            }
            
            return smoothed_prediction, smoothed_confidence, prediction_info
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Prediction error", 0.0, {'error': str(e), 'processing_time': 0.0}
    
    def _robust_normalize(self, X: np.ndarray) -> np.ndarray:
        """Fallback robust normalization"""
        try:
            # Use median and MAD for robust scaling
            median = np.median(X, axis=0)
            mad = np.median(np.abs(X - median), axis=0)
            # Avoid division by zero
            mad[mad == 0] = 1.0
            return (X - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
        except:
            # Ultimate fallback: just return original data
            return X
    
    def _get_top_k_predictions(self, predictions: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Get top K predictions with confidence scores"""
        top_indices = np.argsort(predictions)[-k:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            gesture_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(predictions[idx])
            top_predictions.append({
                'gesture': gesture_name,
                'confidence': confidence
            })
        
        return top_predictions
    
    def _smooth_predictions(self, new_prediction: str, new_confidence: float) -> Tuple[str, float]:
        """Enhanced prediction smoothing with stability tracking"""
        global current_prediction, prediction_confidence, stable_prediction_count, last_stable_prediction
        
        self.prediction_history.append(new_prediction)
        self.confidence_history.append(new_confidence)
        
        # Only consider recent predictions
        if len(self.prediction_history) >= PREDICTION_SMOOTHING_FRAMES:
            recent_predictions = list(self.prediction_history)
            recent_confidences = list(self.confidence_history)
            
            # Count predictions with sufficient confidence
            high_confidence_predictions = {}
            total_confidences = {}
            
            for pred, conf in zip(recent_predictions, recent_confidences):
                if conf > PREDICTION_CONFIDENCE_THRESHOLD:
                    if pred not in high_confidence_predictions:
                        high_confidence_predictions[pred] = 0
                        total_confidences[pred] = 0.0
                    high_confidence_predictions[pred] += 1
                    total_confidences[pred] += conf
            
            # Find most consistent high-confidence prediction
            if high_confidence_predictions:
                # Get prediction with most occurrences
                most_frequent = max(high_confidence_predictions.keys(), 
                                   key=lambda x: high_confidence_predictions[x])
                
                # Check if it meets consensus threshold
                consensus_ratio = high_confidence_predictions[most_frequent] / len(recent_predictions)
                
                if consensus_ratio >= PREDICTION_CONSENSUS_THRESHOLD:
                    avg_confidence = total_confidences[most_frequent] / high_confidence_predictions[most_frequent]
                    
                    # Update stable prediction tracking
                    if most_frequent == self.last_stable_prediction:
                        self.stable_prediction_count += 1
                    else:
                        self.stable_prediction_count = 1
                        self.last_stable_prediction = most_frequent
                    
                    self.current_prediction = most_frequent
                    self.prediction_confidence = avg_confidence
                    
                    return self.current_prediction, self.prediction_confidence
        
        # Fallback to current prediction
        if new_confidence > PREDICTION_CONFIDENCE_THRESHOLD:
            return new_prediction, new_confidence
        else:
            return "Low confidence", new_confidence
    
    def clear_buffers(self):
        """Clear all prediction buffers"""
        self.sequence_buffer.clear()
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.current_prediction = "No hand detected"
        self.prediction_confidence = 0.0
        self.stable_prediction_count = 0

# Initialize enhanced predictor
predictor = EnhancedPredictor(model, scaler, label_encoder, config)

# -------------------------------
# Enhanced Webcam Setup
# -------------------------------
# -------------------------------
# Enhanced Webcam Setup - FIXED VERSION
# -------------------------------
def setup_webcam() -> cv2.VideoCapture:
    """Setup webcam with optimal settings and better error handling"""
    logger.info("Setting up webcam...")
    
    # Try different camera indices and backends
    backends_to_try = [
        cv2.CAP_DSHOW,    # DirectShow (Windows)
        cv2.CAP_MSMF,     # Microsoft Media Foundation (Windows)
        cv2.CAP_V4L2,     # Video4Linux (Linux)
        cv2.CAP_ANY       # Any available backend
    ]
    
    for camera_idx in [0, 1, 2]:
        for backend in backends_to_try:
            logger.info(f"Trying camera {camera_idx} with backend {backend}...")
            
            try:
                cap = cv2.VideoCapture(camera_idx, backend)
                
                if not cap.isOpened():
                    cap.release()
                    continue
                
                # Set buffer size first to avoid frame accumulation
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Try to read a test frame
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    logger.warning(f"Camera {camera_idx} opened but cannot read frames")
                    cap.release()
                    continue
                
                # Check if frame has valid dimensions
                if test_frame.shape[0] < 10 or test_frame.shape[1] < 10:
                    logger.warning(f"Camera {camera_idx} returned invalid frame size: {test_frame.shape}")
                    cap.release()
                    continue
                
                # Configure camera settings after successful test
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Verify settings were applied
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Test one more frame with new settings
                    ret, test_frame2 = cap.read()
                    if ret and test_frame2 is not None:
                        logger.info(f"✓ Webcam {camera_idx} connected successfully!")
                        logger.info(f"  Resolution: {actual_width}x{actual_height}")
                        logger.info(f"  FPS: {actual_fps}")
                        logger.info(f"  Backend: {backend}")
                        return cap
                    
                except Exception as setting_error:
                    logger.warning(f"Could not apply camera settings: {setting_error}")
                    # Still return the camera if basic functionality works
                    if ret and test_frame is not None:
                        logger.info(f"✓ Webcam {camera_idx} connected with default settings")
                        return cap
                
                cap.release()
                
            except Exception as e:
                logger.warning(f"Error with camera {camera_idx}, backend {backend}: {e}")
                try:
                    cap.release()
                except:
                    pass
                continue
    
    logger.error("Could not open any webcam!")
    return None

# Replace the webcam setup section with this:
# Setup webcam with improved error handling
cap = setup_webcam()
if cap is None:
    logger.error("Failed to setup webcam. Exiting.")
    exit(1)

# Get frame dimensions with error handling
frame_width, frame_height = None, None
for attempt in range(3):  # Try up to 3 times
    try:
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            frame_height, frame_width = test_frame.shape[:2]
            logger.info(f"Frame size: {frame_width}x{frame_height}")
            break
        else:
            logger.warning(f"Attempt {attempt + 1}: Could not read test frame")
            time.sleep(0.1)  # Short delay before retry
    except Exception as e:
        logger.warning(f"Attempt {attempt + 1}: Frame read error: {e}")
        time.sleep(0.1)

if frame_width is None or frame_height is None:
    logger.error("Could not determine frame dimensions!")
    cap.release()
    exit(1)

# -------------------------------
# Enhanced UI and Visualization
# -------------------------------
class UIRenderer:
    """Enhanced UI rendering with better visualization"""
    
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # UI colors
        self.colors = {
            'green': (0, 255, 0),
            'yellow': (0, 255, 255), 
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'purple': (255, 0, 255),
            'cyan': (255, 255, 0),
            'gray': (128, 128, 128),
            'dark_gray': (64, 64, 64)
        }
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            fps_end_time = time.time()
            self.current_fps = 30 / (fps_end_time - self.fps_start_time)
            self.fps_start_time = fps_end_time
            self.fps_counter = 0
    
    def draw_hand_landmarks(self, frame: np.ndarray, hand_landmarks, 
                           measurement_positions: Dict[str, Tuple] = None, 
                           best_measurement: str = None):
        """Draw enhanced hand landmarks"""
        # Draw hand connections with better styling
        mp_draw.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_draw_styles.get_default_hand_landmarks_style(),
            mp_draw_styles.get_default_hand_connections_style()
        )
        
        # Draw measurement lines if available
        if measurement_positions:
            # Draw best measurement in bright green
            if best_measurement and best_measurement in measurement_positions:
                start_pos, end_pos = measurement_positions[best_measurement]
                cv2.line(frame, (int(start_pos[0]), int(start_pos[1])), 
                        (int(end_pos[0]), int(end_pos[1])), self.colors['green'], 3)
                
                # Add measurement label
                mid_x = int((start_pos[0] + end_pos[0]) / 2)
                mid_y = int((start_pos[1] + end_pos[1]) / 2) - 10
                cv2.putText(frame, best_measurement.replace('_', ' '), (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['green'], 1)
            
            # Draw other measurements dimmed
            other_colors = [self.colors['blue'], self.colors['cyan'], self.colors['purple']]
            color_idx = 0
            for measurement_name, positions in measurement_positions.items():
                if measurement_name != best_measurement:
                    color = other_colors[color_idx % len(other_colors)]
                    start_pos, end_pos = positions
                    cv2.line(frame, (int(start_pos[0]), int(start_pos[1])), 
                            (int(end_pos[0]), int(end_pos[1])), color, 1)
                    color_idx += 1
    
    def draw_calibration_ui(self, frame: np.ndarray, calibration_frame_count: int, total_frames: int):
        """Draw calibration UI"""
        # Progress bar
        progress = calibration_frame_count / total_frames
        bar_width = 400
        bar_height = 30
        bar_x = (self.frame_width - bar_width) // 2
        bar_y = 50
        
        # Background
        cv2.rectangle(frame, (bar_x - 5, bar_y - 5), 
                     (bar_x + bar_width + 5, bar_y + bar_height + 5), 
                     self.colors['black'], -1)
        
        # Progress fill
        fill_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     self.colors['green'], -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.colors['white'], 2)
        
        # Progress text
        progress_text = f"CALIBRATING... {calibration_frame_count}/{total_frames} ({progress:.1%})"
        text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = (self.frame_width - text_size[0]) // 2
        cv2.putText(frame, progress_text, (text_x, bar_y + bar_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['yellow'], 2)
        
        # Instructions
        instructions = [
            "Hold your hand naturally in front of the camera",
            "Slowly rotate and move your hand in different orientations",
            "Keep your hand visible and well-lit"
        ]
        
        for i, instruction in enumerate(instructions):
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = (self.frame_width - text_size[0]) // 2
            cv2.putText(frame, instruction, (text_x, bar_y + bar_height + 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
    
    def draw_prediction_ui(self, frame: np.ndarray, prediction: str, confidence: float, 
                          prediction_info: Dict[str, Any], hands_detected: int,
                          hand_distance: Optional[float] = None, motion_magnitude: float = 0.0):
        """Draw enhanced prediction UI"""
        # Background box for main prediction
        box_height = 140
        box_width = self.frame_width - 20
        box_x = 10
        box_y = self.frame_height - box_height - 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     self.colors['black'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Main prediction text
        if confidence > PREDICTION_CONFIDENCE_THRESHOLD:
            prediction_display = prediction.replace('_', ' ').title()
            prediction_color = self.colors['green']
        elif confidence > 0.3:
            prediction_display = prediction.replace('_', ' ').title()
            prediction_color = self.colors['yellow']
        else:
            prediction_display = "Low Confidence"
            prediction_color = self.colors['red']
        
        # Large prediction text
        font_scale = 1.0
        prediction_text = f"Gesture: {prediction_display}"
        text_size = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = box_x + (box_width - text_size[0]) // 2
        text_y = box_y + 35
        
        cv2.putText(frame, prediction_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, prediction_color, 2)
        
        # Confidence bar
        confidence_text = f"Confidence: {confidence:.1%}"
        cv2.putText(frame, confidence_text, (box_x + 20, text_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
        
        # Confidence bar visualization
        bar_width = 250
        bar_height = 12
        bar_x = box_x + 20
        bar_y = text_y + 40
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.colors['dark_gray'], -1)
        
        # Confidence fill
        fill_width = int(bar_width * confidence)
        if confidence > PREDICTION_CONFIDENCE_THRESHOLD:
            bar_color = self.colors['green']
        elif confidence > 0.3:
            bar_color = self.colors['yellow']
        else:
            bar_color = self.colors['red']
            
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     bar_color, -1)
        
        # Threshold line
        threshold_x = bar_x + int(bar_width * PREDICTION_CONFIDENCE_THRESHOLD)
        cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), 
                self.colors['white'], 1)
        
        # Top predictions panel
        if 'top_3_predictions' in prediction_info:
            self.draw_top_predictions(frame, prediction_info['top_3_predictions'], 
                                    box_x + bar_width + 40, text_y)
        
        # Performance info
        perf_y = box_y + bar_height + 60
        if 'processing_time' in prediction_info:
            perf_text = f"Processing: {prediction_info['processing_time']*1000:.1f}ms"
            cv2.putText(frame, perf_text, (box_x + 20, perf_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['gray'], 1)
        
        if 'sequence_length' in prediction_info:
            seq_text = f"Buffer: {prediction_info['sequence_length']}/{SEQUENCE_LENGTH}"
            cv2.putText(frame, seq_text, (box_x + 150, perf_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['gray'], 1)
        
        # Hand info
        info_y = perf_y + 15
        if hand_distance is not None:
            distance_text = f"Distance: {hand_distance:.1f}cm"
            cv2.putText(frame, distance_text, (box_x + 20, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['cyan'], 1)
        
        motion_text = f"Motion: {motion_magnitude:.1f} cm/s"
        cv2.putText(frame, motion_text, (box_x + 150, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['cyan'], 1)
    
    def draw_top_predictions(self, frame: np.ndarray, top_predictions: List[Dict[str, Any]], 
                           start_x: int, start_y: int):
        """Draw top predictions panel"""
        panel_width = 200
        panel_height = 80
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y - 10), 
                     (start_x + panel_width, start_y + panel_height), 
                     self.colors['dark_gray'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Top Predictions:", (start_x + 5, start_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['white'], 1)
        
        # Top predictions
        for i, pred_info in enumerate(top_predictions[:3]):
            gesture = pred_info['gesture'].replace('_', ' ')
            conf = pred_info['confidence']
            
            # Color based on confidence
            if conf > 0.6:
                color = self.colors['green']
            elif conf > 0.3:
                color = self.colors['yellow']
            else:
                color = self.colors['red']
            
            pred_text = f"{i+1}. {gesture} ({conf:.1%})"
            cv2.putText(frame, pred_text, (start_x + 5, start_y + 25 + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    def draw_status_bar(self, frame: np.ndarray, hands_detected: int, calibration_mode: bool):
        """Draw status bar"""
        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['green'], 2)
        
        # Hands detected
        hands_color = self.colors['green'] if hands_detected > 0 else self.colors['red']
        cv2.putText(frame, f"Hands: {hands_detected}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hands_color, 2)
        
        # Mode
        mode_text = "CALIBRATING" if calibration_mode else "RECOGNIZING"
        mode_color = self.colors['yellow'] if calibration_mode else self.colors['green']
        cv2.putText(frame, f"Mode: {mode_text}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    
    def draw_gesture_list(self, frame: np.ndarray, available_gestures: List[str], 
                         current_gesture: str):
        """Draw available gestures list"""
        list_x = self.frame_width - 220
        list_y = 100
        
        cv2.putText(frame, "Available Gestures:", (list_x, list_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['white'], 1)
        
        for i, gesture in enumerate(available_gestures[:8]):  # Show first 8
            display_name = gesture.replace('_', ' ').title()
            color = self.colors['green'] if gesture == current_gesture else self.colors['gray']
            cv2.putText(frame, f"• {display_name}", (list_x, list_y + 20 + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

# Initialize UI renderer
ui_renderer = UIRenderer(frame_width, frame_height)

# -------------------------------
# Main Loop
# -------------------------------
def main_loop():
    """Enhanced main processing loop"""
    logger.info("Starting enhanced live gesture recognition...")
    logger.info("Controls:")
    logger.info("  'q' or ESC - Quit")
    logger.info("  'r' - Restart calibration") 
    logger.info("  's' - Skip calibration (if in calibration mode)")
    logger.info("  'c' - Clear prediction buffers")
    
    last_frame_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame, continuing...")
                continue

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_count += 1
            current_time = time.time()
            dt = current_time - last_frame_time
            last_frame_time = current_time
            
            # Update FPS
            ui_renderer.update_fps()
            
            # Process frame timing
            frame_start_time = time.time()
            
            try:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                hands_detected = 0
                current_hand_distance = None
                current_motion_magnitude = 0.0
                
                if results.multi_hand_landmarks and results.multi_handedness:
                    hands_detected = len(results.multi_hand_landmarks)
                    
                    # Process first detected hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    handedness = results.multi_handedness[0]
                    
                    # Calculate palm measurements
                    current_measurements, measurement_positions = feature_extractor.calculate_all_palm_measurements(
                        hand_landmarks.landmark, frame_width, frame_height)
                    
                    # Handle calibration
                    if feature_extractor.calibration_mode:
                        calibration_complete = feature_extractor.update_calibration(current_measurements)
                        ui_renderer.draw_calibration_ui(frame, feature_extractor.calibration_frame_count, 
                                                       feature_extractor.CALIBRATION_FRAMES)
                        
                        # Draw hand landmarks during calibration
                        ui_renderer.draw_hand_landmarks(frame, hand_landmarks, measurement_positions)
                    
                    else:  # Normal recognition mode
                        # Estimate hand distance
                        hand_distance, best_measurement, best_value, ratio = feature_extractor.estimate_hand_distance(current_measurements)
                        current_hand_distance = hand_distance
                        
                        if hand_distance is not None:
                            # Calculate motion features (initial empty ones for comprehensive extraction)
                            initial_motion_features = {
                                'palm_velocity': 0.0,
                                'overall_motion_magnitude': 0.0,
                                'fingertip_velocities': {},
                                'motion_direction': [0.0, 0.0, 0.0],
                                'motion_stability': 1.0
                            }
                            
                            # Extract comprehensive hand features
                            hand_features = feature_extractor.extract_comprehensive_features(
                                hand_landmarks, handedness, hand_distance, frame_width, frame_height, initial_motion_features)
                            
                            # Calculate actual motion features with hand data
                            motion_features = motion_tracker.calculate_motion_features(hand_features, dt)
                            hand_features['motion'] = motion_features
                            current_motion_magnitude = motion_features.get('overall_motion_magnitude', 0.0)
                            
                            # Extract numerical features for prediction
                            frame_data = {'hands': [hand_features]}
                            numerical_features = feature_extractor.extract_frame_features(frame_data)
                            
                            # Make prediction
                            prediction, confidence, prediction_info = predictor.predict_gesture(numerical_features)
                            
                            # Draw enhanced hand landmarks with measurements
                            ui_renderer.draw_hand_landmarks(frame, hand_landmarks, measurement_positions, best_measurement)
                            
                            # Draw prediction UI
                            ui_renderer.draw_prediction_ui(frame, prediction, confidence, prediction_info, 
                                                         hands_detected, hand_distance, current_motion_magnitude)
                        else:
                            # No reliable distance estimation
                            ui_renderer.draw_hand_landmarks(frame, hand_landmarks, measurement_positions)
                            cv2.putText(frame, "Estimating hand distance...", (10, frame_height - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, ui_renderer.colors['yellow'], 2)
                
                else:  # No hands detected
                    hands_detected = 0
                    # Clear prediction buffers when no hands detected
                    predictor.clear_buffers()
                    motion_tracker.previous_features = None
                    
                    if not feature_extractor.calibration_mode:
                        # Show "no hands" message
                        no_hands_text = "No hands detected - Place your hand in view"
                        text_size = cv2.getTextSize(no_hands_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        text_x = (frame_width - text_size[0]) // 2
                        text_y = frame_height // 2
                        
                        # Semi-transparent background
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (text_x - 20, text_y - 30), 
                                     (text_x + text_size[0] + 20, text_y + 10), 
                                     ui_renderer.colors['black'], -1)
                        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                        
                        cv2.putText(frame, no_hands_text, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ui_renderer.colors['yellow'], 2)
                
                # Draw status bar
                ui_renderer.draw_status_bar(frame, hands_detected, feature_extractor.calibration_mode)
                
                # Draw available gestures list (only in recognition mode)
                if not feature_extractor.calibration_mode:
                    available_gestures = list(label_encoder.classes_)
                    current_gesture = predictor.current_prediction if hands_detected > 0 else ""
                    ui_renderer.draw_gesture_list(frame, available_gestures, current_gesture)
                
                # Calculate frame processing time
                frame_processing_time = time.time() - frame_start_time
                frame_processing_times.append(frame_processing_time)
                
                # Show performance info in top right
                if frame_processing_times:
                    avg_frame_time = np.mean(list(frame_processing_times))
                    perf_text = f"Frame: {frame_processing_time*1000:.1f}ms (avg: {avg_frame_time*1000:.1f}ms)"
                    cv2.putText(frame, perf_text, (frame_width - 400, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_renderer.colors['gray'], 1)
                
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                # Draw error message
                error_text = f"Processing Error: {str(e)[:50]}..."
                cv2.putText(frame, error_text, (10, frame_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_renderer.colors['red'], 1)
            
            # Display frame
            cv2.imshow('Enhanced Live Gesture Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                logger.info("Quit requested by user")
                break
                
            elif key == ord('r'):  # Restart calibration
                logger.info("Restarting calibration...")
                feature_extractor.calibration_mode = True
                feature_extractor.calibration_frame_count = 0
                feature_extractor.calibration_data.clear()
                feature_extractor.calibrated_palm_measurements = None
                predictor.clear_buffers()
                motion_tracker.previous_features = None
                
            elif key == ord('s') and feature_extractor.calibration_mode:  # Skip calibration
                if feature_extractor.calibration_frame_count > 10:  # Minimum frames
                    logger.info("Skipping calibration with current data...")
                    # Force completion with current data
                    current_measurements = feature_extractor.calibration_data[-1] if feature_extractor.calibration_data else {}
                    feature_extractor.calibrated_palm_measurements = {}
                    
                    for measurement_name in current_measurements.keys():
                        values = [frame_data[measurement_name] for frame_data in feature_extractor.calibration_data]
                        if values:
                            feature_extractor.calibrated_palm_measurements[measurement_name] = sum(values) / len(values)
                    
                    feature_extractor.calibration_mode = False
                    logger.info("✓ Calibration skipped! Live gesture recognition active.")
                else:
                    logger.warning("Need at least 10 calibration frames before skipping")
                    
            elif key == ord('c'):  # Clear prediction buffers
                logger.info("Clearing prediction buffers...")
                predictor.clear_buffers()
                motion_tracker.previous_features = None
                
            elif key == ord('d'):  # Toggle debug info
                logger.info("Debug toggle requested (feature not implemented)")
                
            elif key == ord('h'):  # Show help
                logger.info("=== HELP ===")
                logger.info("Controls:")
                logger.info("  'q' or ESC - Quit application")
                logger.info("  'r' - Restart calibration process")
                logger.info("  's' - Skip calibration (minimum 10 frames required)")
                logger.info("  'c' - Clear prediction buffers")
                logger.info("  'h' - Show this help")
            
            # Throttle frame rate if needed
            if dt < 1/35:  # Max ~35 FPS
                time.sleep(1/35 - dt)
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        logger.info("✓ Cleanup complete")

# -------------------------------
# Performance Monitoring
# -------------------------------
# -------------------------------
# Performance Monitoring
# -------------------------------
def log_performance_stats():
    """Log performance statistics"""
    if frame_processing_times:
        avg_frame_time = np.mean(list(frame_processing_times))
        max_frame_time = np.max(list(frame_processing_times))
        min_frame_time = np.min(list(frame_processing_times))
        
        logger.info(f"Frame Processing Stats:")
        logger.info(f"  Average: {avg_frame_time*1000:.2f}ms")
        logger.info(f"  Min: {min_frame_time*1000:.2f}ms")
        logger.info(f"  Max: {max_frame_time*1000:.2f}ms")
        logger.info(f"  Effective FPS: {1/avg_frame_time:.1f}")
    
    if predictor.prediction_times:
        avg_pred_time = np.mean(list(predictor.prediction_times))
        logger.info(f"Prediction Processing Stats:")
        logger.info(f"  Average: {avg_pred_time*1000:.2f}ms")
        logger.info(f"  Predictions per second: {1/avg_pred_time:.1f}")

# -------------------------------
# Error Recovery
# -------------------------------
def handle_model_error():
    """Handle model loading errors with recovery suggestions"""
    logger.error("Model loading failed. Recovery suggestions:")
    logger.error("1. Check if the model directory exists: " + MODEL_DIR)
    logger.error("2. Verify model files are present:")
    logger.error(f"   - {MODEL_PATH}")
    logger.error(f"   - {SCALER_PATH}")
    logger.error(f"   - {LABEL_ENCODER_PATH}")
    logger.error("3. Ensure files are not corrupted")
    logger.error("4. Check file permissions")
    logger.error("5. Verify TensorFlow version compatibility")

def handle_camera_error():
    """Handle camera errors with recovery suggestions"""
    logger.error("Camera setup failed. Recovery suggestions:")
    logger.error("1. Check if camera is connected and not in use by another application")
    logger.error("2. Try different camera indices (0, 1, 2)")
    logger.error("3. Check camera permissions")
    logger.error("4. Restart the application")
    logger.error("5. Check if OpenCV is properly installed with camera support")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    try:
        # Verify prerequisites
        logger.info("Verifying prerequisites...")
        
        # Check TensorFlow
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Check OpenCV
        logger.info(f"OpenCV version: {cv2.__version__}")
        
        # Check MediaPipe
        logger.info(f"MediaPipe version: {mp.__version__}")
        
        # Start main loop
        main_loop()
        
        # Log performance stats before exit
        log_performance_stats()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        
        # Provide specific recovery suggestions based on error type
        if "model" in str(e).lower():
            handle_model_error()
        elif "camera" in str(e).lower() or "video" in str(e).lower():
            handle_camera_error()
        else:
            logger.error("General recovery suggestions:")
            logger.error("1. Check all dependencies are installed")
            logger.error("2. Verify file paths and permissions")
            logger.error("3. Check system resources (memory, CPU)")
            logger.error("4. Try restarting the application")
        
        exit(1)
    
    logger.info("Application terminated successfully")

    def calculate_all_palm_measurements(self, landmarks, frame_width: int, frame_height: int) -> Tuple[Dict[str, float], Dict[str, Tuple]]:
        """Calculate comprehensive palm measurements"""
        measurements = {}
        positions = {}
        
        # Key landmarks
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

    def estimate_hand_distance(self, current_measurements: Dict[str, float]) -> Tuple[Optional[float], str, float, float]:
        """Estimate hand distance using calibrated measurements"""
        if self.calibrated_palm_measurements is None:
            return None, "index_to_pinky", current_measurements.get('index_to_pinky', 80), 1.0
        
        # Select best measurement (highest ratio to calibrated)
        best_measurement = None
        best_ratio = 0
        best_current_value = 0
        
        for measurement_name in current_measurements:
            if measurement_name in self.calibrated_palm_measurements:
                current_val = current_measurements[measurement_name]
                calibrated_val = self.calibrated_palm_measurements[measurement_name]
                
                if calibrated_val > 0 and current_val >= 5:  # Minimum threshold
                    ratio = current_val / calibrated_val
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_measurement = measurement_name
                        best_current_value = current_val
        
        if best_measurement is None:
            return None, "index_to_pinky", current_measurements.get('index_to_pinky', 80), 1.0
        
        # Calculate distance based on best measurement
        reference_distance_cm = 40.0
        calibrated_value = self.calibrated_palm_measurements[best_measurement]
        distance_cm = (calibrated_value * reference_distance_cm) / best_current_value
        
        return distance_cm, best_measurement, best_current_value, best_ratio

    def extract_comprehensive_features(self, hand_landmarks, handedness, hand_distance_cm: Optional[float], 
                                     frame_width: int, frame_height: int, motion_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive hand features matching training exactly"""
        features = {
            'hand_label': handedness.classification[0].label,
            'hand_confidence': handedness.classification[0].score,
            'hand_distance_cm': hand_distance_cm if hand_distance_cm is not None else 40.0,
            'palm_measurements': {},
            'landmarks_3d': [],
            'landmarks_normalized': [],
            'fingertip_positions': {},
            'palm_orientation': 0,
            'finger_angles': [],
            'distances': [],
            'motion': motion_features
        }
        
        landmarks = hand_landmarks.landmark
        
        # Calculate palm measurements and orientation
        current_measurements, _ = self.calculate_all_palm_measurements(landmarks, frame_width, frame_height)
        features['palm_measurements'] = current_measurements
        features['palm_orientation'] = self.calculate_palm_orientation(landmarks, frame_width, frame_height)
        
        # Extract all landmark positions with 3D conversion
        focal_length = self.FOCAL_LENGTH_PIXELS
        
        for i, landmark in enumerate(landmarks):
            features['landmarks_normalized'].extend([landmark.x, landmark.y, landmark.z])
            
            # Convert to real-world 3D coordinates
            if hand_distance_cm is not None:
                pixel_x = landmark.x * frame_width
                pixel_y = landmark.y * frame_height
                centered_x = pixel_x - (frame_width / 2)
                centered_y = pixel_y - (frame_height / 2)
                real_x_cm = (centered_x * hand_distance_cm) / focal_length
                real_y_cm = (centered_y * hand_distance_cm) / focal_length
                
                # Enhanced Z calculation for fingertips
                if i in [4, 8, 12, 16, 20]:  # Fingertips
                    real_z_cm = self.calculate_enhanced_fingertip_depth(landmarks, i, hand_distance_cm)
                else:
                    palm_ref_z = self.calculate_palm_plane_reference(landmarks)
                    relative_z = landmark.z - palm_ref_z
                    real_z_cm = hand_distance_cm + (relative_z * hand_distance_cm * 0.2)
                
                features['landmarks_3d'].extend([real_x_cm, real_y_cm, real_z_cm])
            else:
                features['landmarks_3d'].extend([landmark.x * 100, landmark.y * 100, landmark.z * 100])
        
        # Extract fingertip positions
        self.extract_fingertip_positions(features, landmarks, hand_distance_cm, frame_width, frame_height)
        
        # Calculate finger bend angles
        self.calculate_finger_angles(features, landmarks)
        
        # Calculate distances from palm to fingertips
        self.calculate_palm_distances(features, landmarks)
        
        return features

    def calculate_palm_orientation(self, landmarks, frame_width: int, frame_height: int) -> float:
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

    def calculate_palm_plane_reference(self, landmarks) -> float:
        """Calculate reference palm plane using key palm points"""
        palm_landmarks = [0, 1, 5, 9, 13, 17]  # Wrist, Thumb CMC, and all MCP joints
        
        palm_z_sum = sum(landmarks[idx].z for idx in palm_landmarks)
        return palm_z_sum / len(palm_landmarks)

    def calculate_enhanced_fingertip_depth(self, landmarks, fingertip_idx: int, hand_distance_cm: float) -> float:
        """Calculate enhanced fingertip depth"""
        fingertip = landmarks[fingertip_idx]
        palm_reference_z = self.calculate_palm_plane_reference(landmarks)
        relative_z = fingertip.z - palm_reference_z
        z_scale_factor = hand_distance_cm * 0.3
        scaled_relative_z = relative_z * z_scale_factor
        return hand_distance_cm + scaled_relative_z

    def extract_fingertip_positions(self, features: Dict[str, Any], landmarks, hand_distance_cm: Optional[float],
                                  frame_width: int, frame_height: int):
        """Extract fingertip positions"""
        fingertips = [4, 8, 12, 16, 20]
        fingertip_names = ["thumb", "index", "middle", "ring", "pinky"]
        
        for tip_idx, name in zip(fingertips, fingertip_names):
            if hand_distance_cm is not None:
                pixel_x = landmarks[tip_idx].x * frame_width
                pixel_y = landmarks[tip_idx].y * frame_height
                centered_x = pixel_x - (frame_width / 2)
                centered_y = pixel_y - (frame_height / 2)
                real_x_cm = (centered_x * hand_distance_cm) / self.FOCAL_LENGTH_PIXELS
                real_y_cm = (centered_y * hand_distance_cm) / self.FOCAL_LENGTH_PIXELS
                real_z_cm = self.calculate_enhanced_fingertip_depth(landmarks, tip_idx, hand_distance_cm)
                
                features['fingertip_positions'][name] = {
                    'x': real_x_cm, 'y': real_y_cm, 'z': real_z_cm
                }
            else:
                features['fingertip_positions'][name] = {
                    'x': landmarks[tip_idx].x * 10, 
                    'y': landmarks[tip_idx].y * 10, 
                    'z': landmarks[tip_idx].z * 10
                }

    def calculate_finger_angles(self, features: Dict[str, Any], landmarks):
        """Calculate finger bend angles"""
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
                    cos_angle = max(-1, min(1, dot_product / (mag1 * mag2)))
                    angle = math.acos(cos_angle)
                    features['finger_angles'].append(angle)
                else:
                    features['finger_angles'].append(0)

    def calculate_palm_distances(self, features: Dict[str, Any], landmarks):
        """Calculate distances from palm center to fingertips"""
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

    def extract_frame_features(self, frame_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from frame matching training exactly"""
        features = np.zeros(self.config.feature_dim, dtype=np.float32)

        hands = frame_data.get('hands', [])
        if not hands:
            return features

        try:
            hand_data = hands[0]

            # Basic hand info (3 features)
            features[0] = hand_data.get('hand_confidence', 0.0)
            features[1] = hand_data.get('hand_distance_cm', 40.0)
            features[2] = hand_data.get('palm_orientation', 0.0)

            # Palm measurements (3 features)
            palm_measurements = hand_data.get('palm_measurements', {})
            features[3] = palm_measurements.get('index_to_pinky', 80.0)
            features[4] = palm_measurements.get('thumb_to_pinky', 120.0)
            features[5] = palm_measurements.get('wrist_to_mcp', 90.0)

            # Landmarks 3D (63 features)
            landmarks_3d = hand_data.get('landmarks_3d', [])
            if landmarks_3d and len(landmarks_3d) >= 63:
                features[6:69] = landmarks_3d[:63]

            # Finger angles (5 features)
            finger_angles = hand_data.get('finger_angles', [])
            if finger_angles and len(finger_angles) >= 5:
                features[69:74] = finger_angles[:5]

            # Distances (5 features)
            distances = hand_data.get('distances', [])
            if distances and len(distances) >= 5:
                features[74:79] = distances[:5]

            # Motion features (4 features)
            motion = hand_data.get('motion', {})
            features[79] = motion.get('palm_velocity', 0.0)
            features[80] = motion.get('overall_motion_magnitude', 0.0)
            features[81] = len(motion.get('fingertip_velocities', {}))
            motion_direction = motion.get('motion_direction', [0, 0, 0])
            features[82] = abs(sum(motion_direction)) if motion_direction else 0.0

            # Fingertip positions (15 features)
            fingertip_positions = hand_data.get('fingertip_positions', {})
            fingertip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            for i, finger_name in enumerate(fingertip_names):
                base_idx = 83 + i * 3
                if finger_name in fingertip_positions:
                    pos = fingertip_positions[finger_name]
                    features[base_idx] = pos.get('x', 0.0)
                    features[base_idx + 1] = pos.get('y', 0.0)
                    features[base_idx + 2] = pos.get('z', 40.0)

        except Exception as e:
            logger.warning(f"Error extracting frame features: {e}")

        return features

    def update_calibration(self, current_measurements: Dict[str, float]) -> bool:
        """Update calibration with current measurements"""
        if not self.calibration_mode:
            return False
            
        self.calibration_frame_count += 1
        self.calibration_data.append(current_measurements.copy())
        
        if self.calibration_frame_count >= self.CALIBRATION_FRAMES:
            # Complete calibration
            self.calibrated_palm_measurements = {}
            for measurement_name in current_measurements.keys():
                values = [frame_data[measurement_name] for frame_data in self.calibration_data]
                self.calibrated_palm_measurements[measurement_name] = sum(values) / len(values)
            
            self.calibration_mode = False
            logger.info("✓ Calibration complete! Live gesture recognition active.")
            return True
        return False

# Initialize feature extractor
feature_extractor = FeatureExtractor(config)

# -------------------------------
# Enhanced Motion Tracking
# -------------------------------
class MotionTracker:
    """Enhanced motion tracking for gesture recognition"""
    
    def __init__(self, smoothing_factor: float = 0.7):
        self.previous_features = None
        self.smoothing_factor = smoothing_factor
        self.velocity_history = deque(maxlen=5)
        
    def calculate_motion_features(self, current_features: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """Calculate enhanced motion features"""
        motion_features = {
            'palm_velocity': 0.0,
            'palm_acceleration': 0.0,
            'fingertip_velocities': {},
            'overall_motion_magnitude': 0.0,
            'motion_direction': [0.0, 0.0, 0.0],
            'motion_stability': 1.0
        }
        
        if self.previous_features is None or dt <= 0:
            self.previous_features = current_features
            return motion_features
        
        try:
            # Calculate palm motion
            if ('landmarks_3d' in current_features and 'landmarks_3d' in self.previous_features and
                len(current_features['landmarks_3d']) >= 3 and len(self.previous_features['landmarks_3d']) >= 3):
                
                current_wrist = current_features['landmarks_3d'][0:3]
                previous_wrist = self.previous_features['landmarks_3d'][0:3]
                
                velocity_vector = [(curr - prev) / dt for curr, prev in zip(current_wrist, previous_wrist)]
                palm_velocity = math.sqrt(sum(v*v for v in velocity_vector))
                
                # Apply smoothing
                if self.velocity_history:
                    palm_velocity = self.smoothing_factor * palm_velocity + (1 - self.smoothing_factor) * self.velocity_history[-1]
                
                self.velocity_history.append(palm_velocity)
                
                motion_features['palm_velocity'] = palm_velocity
                motion_features['motion_direction'] = velocity_vector
            
            # Calculate fingertip velocities
            velocities = []
            for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                if (finger_name in current_features['fingertip_positions'] and 
                    finger_name in self.previous_features['fingertip_positions']):
                    
                    curr_pos = current_features['fingertip_positions'][finger_name]
                    prev_pos = self.previous_features['fingertip_positions'][finger_name]
                    
                    dx = (curr_pos['x'] - prev_pos['x']) / dt
                    dy = (curr_pos['y'] - prev_pos['y']) / dt
                    dz = (curr_pos['z'] - prev_pos['z']) / dt
                    
                    velocity = math.sqrt(dx*dx + dy*dy + dz*dz)
                    velocities.append(velocity)
                    
                    motion_features['fingertip_velocities'][finger_name] = {
                        'velocity': velocity,
                        'velocity_vector': [dx, dy, dz]
                    }
            
            # Calculate overall motion magnitude
            all_velocities = [motion_features['palm_velocity']] + velocities
            if all_velocities:
                motion_features['overall_motion_magnitude'] = sum(all_velocities) / len(all_velocities)
            
            # Calculate motion stability (inverse of velocity variance)
            if len(self.velocity_history) >= 3:
                velocity_variance = np.var(list(self.velocity_history))
                motion_features['motion_stability'] = 1.0 / (1.0 + velocity_variance)
        
        except Exception as e:
            logger.warning(f"Error calculating motion features: {e}")
        
        self.previous_features = current_features.copy()
        return motion_features

# Initialize motion tracker
motion_tracker = MotionTracker()

# -------------------------------
# Enhanced Prediction System
# -------------------------------
