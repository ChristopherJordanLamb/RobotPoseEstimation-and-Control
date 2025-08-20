import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Attention
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import warnings
from datetime import datetime
import yaml

warnings.filterwarnings('ignore')

@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    data_dir: str = "gesture_training_data"
    model_save_path: str = "gesture_model.h5"
    scaler_save_path: str = "gesture_scaler.pkl" 
    label_encoder_save_path: str = "gesture_label_encoder.pkl"
    
    # Model parameters
    max_sequence_length: int = 150
    feature_dim: int = 98
    
    # Training parameters
    test_size: float = 0.2
    validation_size: float = 0.2    
    batch_size: int = 16
    epochs: int = 1000
    patience: int = 40
    learning_rate: float = 0.001
    min_samples_per_class: int = 3
    
    # Augmentation parameters
    noise_factor: float = 0.02
    time_stretch_factor: float = 0.1
    
    # Cross-validation
    use_cross_validation: bool = False
    cv_folds: int = 5
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        return cls()
    
    def save_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

class DataAugmentation:
    """Data augmentation utilities for gesture sequences"""
    
    @staticmethod
    def add_noise(sequence: np.ndarray, noise_factor: float = 0.02) -> np.ndarray:
        """Add Gaussian noise to sequence"""
        noise = np.random.normal(0, noise_factor, sequence.shape)
        return sequence + noise
    
    @staticmethod
    def time_stretch(sequence: np.ndarray, factor_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Apply time stretching to sequence"""
        factor = np.random.uniform(*factor_range)
        original_length = len(sequence)
        new_length = int(original_length * factor)
        
        if new_length < 1:
            return sequence
            
        indices = np.linspace(0, original_length - 1, new_length)
        stretched = np.array([sequence[int(i)] for i in indices])
        return stretched
    
    @staticmethod
    def augment_sequence(sequence: np.ndarray, config: TrainingConfig) -> List[np.ndarray]:
        """Apply multiple augmentation techniques"""
        augmented = [sequence]  # Original sequence
        
        # Add noise
        augmented.append(DataAugmentation.add_noise(sequence, config.noise_factor))
        
        # Time stretch
        stretched = DataAugmentation.time_stretch(sequence, (1-config.time_stretch_factor, 1+config.time_stretch_factor))
        augmented.append(stretched)
        
        return augmented

class GestureDataLoader:
    """Enhanced gesture data loader with better error handling and feature extraction"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self) -> List[str]:
        """Get feature names for better interpretability"""
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
        
    def extract_features_from_frame(self, frame_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from a single frame with better error handling"""
        features = np.zeros(self.config.feature_dim, dtype=np.float32)

        hands = frame_data.get('hands', [])
        if not hands:
            return features

        try:
            hand_data = hands[0]  # Use only the first hand

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
            if landmarks_3d:
                end_idx = min(len(landmarks_3d), 63)
                features[6:6+end_idx] = landmarks_3d[:end_idx]

            # Finger angles (5 features)
            finger_angles = hand_data.get('finger_angles', [])
            if finger_angles:
                end_idx = min(len(finger_angles), 5)
                features[69:69+end_idx] = finger_angles[:end_idx]

            # Distances (5 features)
            distances = hand_data.get('distances', [])
            if distances:
                end_idx = min(len(distances), 5)
                features[74:74+end_idx] = distances[:end_idx]

            # Motion features (4 features)
            motion = hand_data.get('motion', {})
            features[79] = motion.get('palm_velocity', 0.0)
            features[80] = motion.get('overall_motion_magnitude', 0.0)
            features[81] = len(motion.get('fingertip_velocities', {}))
            motion_direction = motion.get('motion_direction', [0, 0, 0])
            features[82] = abs(sum(motion_direction))

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
            logger.warning(f"Error extracting features from frame: {e}")

        return features

    def load_gesture_file(self, filepath: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Load and process a single gesture file with better error handling"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            gesture_label = data.get('label')
            if not gesture_label:
                logger.warning(f"No label found in {filepath}")
                return None, None
            
            sequence_data = data.get('sequence', [])
            if not sequence_data:
                logger.warning(f"No sequence data found in {filepath}")
                return None, None
            
            sequence = []
            for frame in sequence_data:
                frame_features = self.extract_features_from_frame(frame)
                sequence.append(frame_features)
            
            if not sequence:
                logger.warning(f"No valid frames found in {filepath}")
                return None, None
            
            logger.debug(f"Processed {len(sequence)} frames for {gesture_label} from {filepath.name}")
            return np.array(sequence, dtype=np.float32), gesture_label
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None, None
    
    def load_all_data(self) -> Tuple[Optional[List[np.ndarray]], Optional[List[str]]]:
        """Load all gesture data from the directory with progress tracking"""
        data_path = Path(self.config.data_dir)
        logger.info(f"Loading gesture data from {data_path}...")
        
        if not data_path.exists():
            logger.error(f"Directory {data_path} does not exist!")
            return None, None
        
        gesture_files = list(data_path.glob('*.json'))
        gesture_files = [f for f in gesture_files if f.name != 'summary.json']
        
        if not gesture_files:
            logger.error("No gesture files found!")
            return None, None
        
        logger.info(f"Found {len(gesture_files)} gesture files")
        
        X_data, y_data = [], []
        successful_loads = 0
        
        for filepath in gesture_files:
            sequence, label = self.load_gesture_file(filepath)
            
            if sequence is not None and label is not None:
                X_data.append(sequence)
                y_data.append(label)
                successful_loads += 1
        
        logger.info(f"Successfully loaded {successful_loads}/{len(gesture_files)} files")
        
        # Print dataset statistics
        if y_data:
            label_counts = Counter(y_data)
            logger.info("Dataset composition:")
            for label, count in sorted(label_counts.items()):
                logger.info(f"  {label}: {count} samples")
        
        return X_data, y_data

    def preprocess_sequences(self, X_data: List[np.ndarray]) -> np.ndarray:
        """Enhanced sequence preprocessing with relative positioning"""
        logger.info(f"Preprocessing {len(X_data)} sequences (max length: {self.config.max_sequence_length})...")

        seq_lengths = [len(seq) for seq in X_data]
        logger.info(f"Sequence length stats: Min={min(seq_lengths)}, Max={max(seq_lengths)}, Avg={np.mean(seq_lengths):.1f}")

        processed_sequences = []

        for sequence in X_data:
            sequence = np.array(sequence, dtype=np.float32)

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

        logger.info(f"Final preprocessed data shape: {X_padded.shape}")
        return X_padded

    def _apply_relative_positioning(self, sequence: np.ndarray) -> np.ndarray:
        """Apply relative positioning based on first frame"""
        if len(sequence) == 0:
            return sequence
            
        base_frame = sequence[0].copy()
        
        # Landmark indices (3D coordinates)
        landmark_indices = list(range(6, 69))  # 21 landmarks * 3 coords
        fingertip_indices = list(range(83, 98))  # 5 fingertips * 3 coords
        
        # Apply translation relative to first frame for 3D coordinates
        for idx_group in [landmark_indices, fingertip_indices]:
            for j in range(0, len(idx_group), 3):  # Process x, y, z coordinates
                if j + 2 < len(idx_group):
                    sequence[:, idx_group[j]]     -= base_frame[idx_group[j]]      # x
                    sequence[:, idx_group[j + 1]] -= base_frame[idx_group[j + 1]]  # y
                    sequence[:, idx_group[j + 2]] -= base_frame[idx_group[j + 2]]  # z

        # Normalize hand distance relative to first frame
        sequence[:, 1] -= base_frame[1]  # hand_distance_cm
        
        return sequence

    def prepare_data(self, apply_augmentation: bool = True) -> Tuple[np.ndarray, ...]:
        """Enhanced data preparation with augmentation and better splitting"""
        # Load raw data
        X_data, y_data = self.load_all_data()
        
        if X_data is None or y_data is None:
            raise ValueError("Failed to load data")

        # Filter underrepresented classes
        label_counts = Counter(y_data)
        valid_indices = [i for i, label in enumerate(y_data) 
                        if label_counts[label] >= self.config.min_samples_per_class]

        if not valid_indices:
            raise ValueError("No classes have enough samples")

        X_data = [X_data[i] for i in valid_indices]
        y_data = [y_data[i] for i in valid_indices]

        logger.info("Classes after filtering:")
        filtered_counts = Counter(y_data)
        for label, count in sorted(filtered_counts.items()):
            logger.info(f"  {label}: {count} samples")

        # Apply data augmentation before preprocessing
        if apply_augmentation:
            X_data, y_data = self._apply_augmentation(X_data, y_data)

        # Preprocess sequences
        X_processed = self.preprocess_sequences(X_data)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_data)
        y_categorical = to_categorical(y_encoded)

        logger.info("Label encoding:")
        for i, label in enumerate(self.label_encoder.classes_):
            logger.info(f"  {i}: {label}")

        # Normalize features
        logger.info("Normalizing features...")
        n_samples, n_timesteps, n_features = X_processed.shape
        X_reshaped = X_processed.reshape(-1, n_features)
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X_final = X_normalized.reshape(n_samples, n_timesteps, n_features)

        # Split data (stratified)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_final, y_categorical,
            test_size=self.config.test_size + self.config.validation_size,
            random_state=42,
            stratify=y_encoded
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.config.test_size / (self.config.test_size + self.config.validation_size),
            random_state=42,
            stratify=np.argmax(y_temp, axis=1)
        )

        # Balance training set only
        X_train, y_train = self._balance_dataset(X_train, y_train)

        logger.info(f"Final data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _apply_augmentation(self, X_data: List[np.ndarray], y_data: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """Apply data augmentation to increase dataset size"""
        logger.info("Applying data augmentation...")
        
        augmented_X, augmented_y = [], []
        
        for sequence, label in zip(X_data, y_data):
            augmented_sequences = DataAugmentation.augment_sequence(sequence, self.config)
            for aug_seq in augmented_sequences:
                augmented_X.append(aug_seq)
                augmented_y.append(label)
        
        logger.info(f"Augmentation increased dataset from {len(X_data)} to {len(augmented_X)} samples")
        return augmented_X, augmented_y

    def _balance_dataset(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance the training dataset using oversampling"""
        logger.info("Balancing training dataset...")
        
        y_train_labels = np.argmax(y_train, axis=1)
        unique_labels, counts = np.unique(y_train_labels, return_counts=True)
        max_count = max(counts)
        
        X_balanced, y_balanced = [], []
        
        for label in unique_labels:
            mask = y_train_labels == label
            X_label = X_train[mask]
            y_label = y_train[mask]
            
            # Oversample to match max count
            n_samples = len(X_label)
            if n_samples < max_count:
                indices = np.random.choice(n_samples, size=max_count, replace=True)
                X_resampled = X_label[indices]
                y_resampled = y_label[indices]
            else:
                X_resampled = X_label
                y_resampled = y_label
                
            X_balanced.extend(X_resampled)
            y_balanced.extend(y_resampled)
        
        X_balanced = np.array(X_balanced)
        y_balanced = np.array(y_balanced)
        
        logger.info(f"Balanced training set shape: {X_balanced.shape}")
        return X_balanced, y_balanced

class ModelBuilder:
    """Enhanced model builder with attention mechanism and better architecture"""
    
    @staticmethod
    def create_model(input_shape: Tuple[int, int], num_classes: int, config: TrainingConfig) -> tf.keras.Model:
        """Create an improved gesture recognition model with attention"""
        logger.info(f"Creating model - Input shape: {input_shape}, Classes: {num_classes}")
        
        model = Sequential([
            # Feature extraction layers
            Conv1D(64, kernel_size=7, activation='relu', input_shape=input_shape, padding='same'),
            BatchNormalization(),
            Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            Conv1D(128, kernel_size=5, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Temporal modeling with LSTM
            LSTM(256, return_sequences=True, dropout=0.3),
            LSTM(128, return_sequences=True, dropout=0.3), 
            LSTM(64, return_sequences=False, dropout=0.3),
            
            # Classification layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile with advanced optimizer settings
        optimizer = Adam(
            learning_rate=config.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        logger.info(f"Model created with {model.count_params():,} parameters")
        model.summary(print_fn=logger.info)
        
        return model

class GestureTrainer:
    """Main training class with enhanced features"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = Path(f"models/20250818_234215")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Setup training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.model_dir / "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            TensorBoard(
                log_dir=str(self.model_dir / "logs"),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        return callbacks
    
    def train(self) -> Tuple[tf.keras.Model, GestureDataLoader]:
        """Enhanced training pipeline"""
        logger.info("=" * 50)
        logger.info("Starting Enhanced Gesture Recognition Training")
        logger.info("=" * 50)
        
        # Save configuration
        self.config.save_yaml(str(self.model_dir / "config.yaml"))
        
        # Load and prepare data
        data_loader = GestureDataLoader(self.config)
        
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data()
        except ValueError as e:
            logger.error(f"Data preparation failed: {e}")
            return None, None
        
        # Create model
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        model = ModelBuilder.create_model(input_shape, num_classes, self.config)
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        # Train model
        logger.info(f"Training model for up to {self.config.epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model weights
        model.load_weights(str(self.model_dir / "best_model.h5"))
        
        # Final evaluation
        self._evaluate_model(model, X_test, y_test, data_loader)
        
        # Save final components
        self._save_model_components(model, data_loader)
        
        # Generate visualizations
        self._create_visualizations(history, model, X_test, y_test, data_loader)
        
        return model, data_loader
    
    def _evaluate_model(self, model: tf.keras.Model, X_test: np.ndarray, 
                       y_test: np.ndarray, data_loader: GestureDataLoader):
        """Comprehensive model evaluation"""
        logger.info("Evaluating model on test set...")
        
        # Basic evaluation
        test_loss, test_accuracy, test_top_k = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Results:")
        logger.info(f"  Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Top-k Accuracy: {test_top_k:.4f}")
        logger.info(f"  Loss: {test_loss:.4f}")
        
        # Detailed predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Classification report
        class_names = data_loader.label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=class_names)
        logger.info(f"\nClassification Report:\n{report}")
        
        # Save evaluation results
        results = {
            'test_accuracy': float(test_accuracy),
            'test_top_k_accuracy': float(test_top_k),
            'test_loss': float(test_loss),
            'classification_report': report,
            'model_parameters': int(model.count_params())
        }
        
        with open(self.model_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    def _save_model_components(self, model: tf.keras.Model, data_loader: GestureDataLoader):
        """Save all model components"""
        logger.info("Saving model components...")
        
        # Save model
        model.save(str(self.model_dir / "final_model.h5"))
        
        # Save preprocessing components
        with open(self.model_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(data_loader.scaler, f)
            
        with open(self.model_dir / "label_encoder.pkl", 'wb') as f:
            pickle.dump(data_loader.label_encoder, f)
        
        # Save feature names for interpretability
        with open(self.model_dir / "feature_names.json", 'w') as f:
            json.dump(data_loader.feature_names, f, indent=2)
        
        logger.info(f"All components saved to {self.model_dir}")
    
    def _create_visualizations(self, history, model: tf.keras.Model, X_test: np.ndarray,
                             y_test: np.ndarray, data_loader: GestureDataLoader):
        """Create and save visualizations"""
        logger.info("Creating visualizations...")
        
        # Training history
        self._plot_training_history(history)
        
        # Confusion matrix
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        y_true = np.argmax(y_test, axis=1)
        self._plot_confusion_matrix(y_true, y_pred, data_loader.label_encoder.classes_)
        
        logger.info(f"Visualizations saved to {self.model_dir}")
    
    def _plot_training_history(self, history):
        """Enhanced training history visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plots
        ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plots
        ax2.plot(history.history['loss'], label='Training', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Top-k accuracy
        ax3.plot(history.history['top_k_categorical_accuracy'], label='Training', linewidth=2)
        if 'val_top_k_categorical_accuracy' in history.history:
            ax3.plot(history.history['val_top_k_categorical_accuracy'], label='Validation', linewidth=2)
            ax3.set_title('Top-K Categorical Accuracy', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Top-K Acc')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Learning rate (if logged)
            # Keras History doesn't include LR by default; ReduceLROnPlateau changes it internally.
            # We'll try to read from optimizer if present per epoch; otherwise leave blank.
            if 'lr' in history.history:
                ax4.plot(history.history['lr'], label='Learning Rate', linewidth=2)
                ax4.set_ylabel('LR')
            else:
                # Fallback: show val_loss again but annotate that LR not tracked
                ax4.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
                ax4.set_ylabel('Loss')
            ax4.set_title('Learning Rate (or Val Loss if LR unavailable)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            out_path = self.model_dir / "training_history.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved training history plot to {out_path}")

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray):
        """Plot and save a confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = self.model_dir / "confusion_matrix.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved confusion matrix to {out_path}")


def main():
    # Optionally load config from YAML if present in CWD (config.yaml), else defaults
    default_cfg_path = "gesture_training_config.yaml"
    config = TrainingConfig.from_yaml(default_cfg_path) if os.path.exists(default_cfg_path) else TrainingConfig()

    # Warn if CV requested but not implemented in this pipeline split
    if config.use_cross_validation:
        logger.warning("use_cross_validation=True is set, but current pipeline uses a single stratified split. "
                    "Proceeding with single split training.")

    trainer = GestureTrainer(config)
    model, data_loader = trainer.train()
    if model is None or data_loader is None:
        logger.error("Training did not complete due to earlier errors.")
        return

    # Save a handy label map next to the final artifacts in root for quick load
    try:
        with open(trainer.model_dir / "label_map.json", "w") as f:
            json.dump({int(i): lbl for i, lbl in enumerate(data_loader.label_encoder.classes_)}, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write label_map.json: {e}")

    logger.info("All done.")


if __name__ == "__main__":
    main()
