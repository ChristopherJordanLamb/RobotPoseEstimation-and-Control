import os
import json
import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Gesture Recognition Model Trainer")
print("="*50)

# Configuration
DATA_DIR = "gesture_training_data"
MODEL_SAVE_PATH = "gesture_model.h5"
SCALER_SAVE_PATH = "gesture_scaler.pkl"
LABEL_ENCODER_SAVE_PATH = "gesture_label_encoder.pkl"

# Training parameters
MAX_SEQUENCE_LENGTH = 150  # Limit sequence length for memory efficiency
FEATURE_DIM = 100  # Estimated feature dimension (we'll adjust based on actual data)
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 200
PATIENCE = 40
class GestureDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def extract_features_from_frame(self, frame_data):
        """Extract numerical features from a single frame using only the first hand"""
        features = []

        hands = frame_data.get('hands', [])
        if hands:
            hand_data = hands[0]  # take only the first hand

            # Basic hand info
            features.extend([
                hand_data.get('hand_confidence', 0),
                hand_data.get('hand_distance_cm', 40),  # Default distance
                hand_data.get('palm_orientation', 0)
            ])

            # Palm measurements (3 values)
            palm_measurements = hand_data.get('palm_measurements', {})
            features.extend([
                palm_measurements.get('index_to_pinky', 80),
                palm_measurements.get('thumb_to_pinky', 120),
                palm_measurements.get('wrist_to_mcp', 90)
            ])

            # Landmarks 3D (21 landmarks * 3 coordinates = 63 values)
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
                len(motion.get('fingertip_velocities', {})),  # Number of moving fingertips
                abs(sum(motion.get('motion_direction', [0, 0, 0])))  # Total motion vector magnitude
            ])

            # Fingertip positions (5 fingertips * 3 coordinates = 15 values)
            fingertip_positions = hand_data.get('fingertip_positions', {})
            fingertip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            for finger_name in fingertip_names:
                if finger_name in fingertip_positions:
                    pos = fingertip_positions[finger_name]
                    features.extend([pos.get('x', 0), pos.get('y', 0), pos.get('z', 40)])
                else:
                    features.extend([0, 0, 40])
        else:
            # If no hands detected, return zero features
            features = [0] * 98

        return features

    def load_gesture_file(self, filepath):
        """Load and process a single gesture file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            gesture_label = data['label']
            sequence = []
            
            print(f"  Processing {len(data['sequence'])} frames for {gesture_label}")
            
            for frame in data['sequence']:
                frame_features = self.extract_features_from_frame(frame)
                sequence.append(frame_features)
            
            return np.array(sequence), gesture_label
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def load_all_data(self):
        """Load all gesture data from the directory"""
        print(f"Loading gesture data from {self.data_dir}...")
        
        X_data = []
        y_data = []
        
        if not os.path.exists(self.data_dir):
            print(f"Error: Directory {self.data_dir} does not exist!")
            return None, None
        
        gesture_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json') and f != 'summary.json']
        
        if not gesture_files:
            print("No gesture files found!")
            return None, None
        
        print(f"Found {len(gesture_files)} gesture files")
        
        for filename in gesture_files:
            filepath = os.path.join(self.data_dir, filename)
            print(f"Loading: {filename}")
            
            sequence, label = self.load_gesture_file(filepath)
            
            if sequence is not None and label is not None:
                X_data.append(sequence)
                y_data.append(label)
        
        print(f"\nLoaded {len(X_data)} gesture sequences")
        
        # Print dataset statistics
        label_counts = Counter(y_data)
        print("\nDataset composition:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples")
        
        return X_data, y_data
    def preprocess_sequences(self, X_data, max_length=MAX_SEQUENCE_LENGTH):
        """Preprocess, normalize relative positions, translate sequences to common base, and pad"""
        print(f"\nPreprocessing sequences (max length: {max_length})...")

        seq_lengths = [len(seq) for seq in X_data]
        print(f"Sequence length stats:")
        print(f"  Min: {min(seq_lengths)} frames")
        print(f"  Max: {max(seq_lengths)} frames")
        print(f"  Average: {np.mean(seq_lengths):.1f} frames")

        processed_sequences = []

        for i, sequence in enumerate(X_data):
            sequence = np.array(sequence, dtype=np.float32)

            # Truncate if too long
            if len(sequence) > max_length:
                start_idx = (len(sequence) - max_length) // 2
                sequence = sequence[start_idx:start_idx + max_length]

            if len(sequence) > 0:
                base_frame = sequence[0].copy()

                # --- Apply translation relative to first frame ---
                landmark_indices = np.arange(6, 69)     # landmarks 3D
                fingertip_indices = np.arange(83, 98)   # fingertips 3D

                for idx_group in [landmark_indices, fingertip_indices]:
                    for j in range(0, len(idx_group), 3):
                        sequence[:, idx_group[j]]   -= base_frame[idx_group[j]]
                        sequence[:, idx_group[j+1]] -= base_frame[idx_group[j+1]]
                        sequence[:, idx_group[j+2]] -= base_frame[idx_group[j+2]]

                # Normalize hand distance
                sequence[:, 1] -= base_frame[1]

            processed_sequences.append(sequence)

        # Pad sequences to same length
        X_padded = tf.keras.preprocessing.sequence.pad_sequences(
            processed_sequences,
            maxlen=max_length,
            dtype='float32',
            padding='post'
        )

        print(f"Final data shape: {X_padded.shape}")
        return X_padded


    def prepare_data(self):
        """Load and prepare all data for training"""
        # Load raw data
        X_data, y_data = self.load_all_data()
        
        if X_data is None or y_data is None:
            return None, None, None, None, None, None

        # Filter out underrepresented classes
        min_samples = 3
        label_counts = Counter(y_data)
        valid_indices = [i for i, label in enumerate(y_data) if label_counts[label] >= min_samples]

        X_data = [X_data[i] for i in valid_indices]
        y_data = [y_data[i] for i in valid_indices]

        print("\nUsing classes after filtering:")
        filtered_counts = Counter(y_data)
        for label, count in filtered_counts.items():
            print(f"  {label}: {count} samples")

        # Preprocess sequences
        X_processed = self.preprocess_sequences(X_data)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_data)
        y_categorical = to_categorical(y_encoded)

        print(f"\nLabel encoding:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {label}")

        # Normalize features
        print("\nNormalizing features...")
        n_samples, n_timesteps, n_features = X_processed.shape
        X_reshaped = X_processed.reshape(-1, n_features)
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X_final = X_normalized.reshape(n_samples, n_timesteps, n_features)

        # -----------------------------
        # Split data first (no leakage)
        # -----------------------------
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_final, y_categorical,
            test_size=TEST_SIZE + VALIDATION_SIZE,
            random_state=42,
            stratify=y_encoded
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE),
            random_state=42,
            stratify=np.argmax(y_temp, axis=1)
        )

        # -----------------------------
        # Oversample training set only
        # -----------------------------
        from sklearn.utils import resample

        X_train_bal, y_train_bal = [], []
        y_train_labels = np.argmax(y_train, axis=1)
        max_count = max(np.bincount(y_train_labels))

        for label in np.unique(y_train_labels):
            X_label = [X_train[i] for i in range(len(X_train)) if y_train_labels[i] == label]
            y_label = [y_train[i] for i in range(len(y_train)) if y_train_labels[i] == label]

            X_res, y_res = resample(
                X_label, y_label,
                replace=True,
                n_samples=max_count,
                random_state=42
            )

            X_train_bal.extend(X_res)
            y_train_bal.extend(y_res)

        X_train = np.array(X_train_bal)
        y_train = np.array(y_train_bal)

        print(f"\nBalanced training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

def create_model(input_shape, num_classes):
    """Create the gesture recognition model"""
    print(f"\nCreating model for input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    model = Sequential([
        # Conv1D layers for feature extraction
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # LSTM layers for temporal modeling
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        # Dense layers for classification
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model

def f1_score_macro(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)

    # Calculate per-class F1
    f1s = []
    num_classes = tf.reduce_max(y_true) + 1
    for i in range(num_classes):
        y_true_i = tf.cast(tf.equal(y_true, i), tf.float32)
        y_pred_i = tf.cast(tf.equal(y_pred, i), tf.float32)

        tp = tf.reduce_sum(y_true_i * y_pred_i)
        fp = tf.reduce_sum((1 - y_true_i) * y_pred_i)
        fn = tf.reduce_sum(y_true_i * (1 - y_pred_i))

        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        f1s.append(f1)
    
    macro_f1 = tf.reduce_mean(f1s)
    return macro_f1
def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Starting gesture recognition model training...")
    
    # Load and prepare data
    data_loader = GestureDataLoader(DATA_DIR)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data()
    
    if X_train is None:
        print("Failed to load data. Exiting.")
        return
    all_classes = set(data_loader.label_encoder.classes_)
    test_classes = set(data_loader.label_encoder.inverse_transform(np.unique(np.argmax(y_test, axis=1))))
    missing_classes = all_classes - test_classes
    print("Missing classes in test set:", missing_classes)
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]
    model = create_model(input_shape, num_classes)
    
    # Define callbacks
    callbacks = [
        # EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Train model
    print(f"\nTraining model for up to {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    model.load_weights(MODEL_SAVE_PATH)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Predictions and detailed evaluation
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Classification report
    class_names = data_loader.label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save model components
    print("\nSaving model components...")
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(data_loader.scaler, f)
    
    with open(LABEL_ENCODER_SAVE_PATH, 'wb') as f:
        pickle.dump(data_loader.label_encoder, f)
    
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Scaler saved to: {SCALER_SAVE_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_SAVE_PATH}")
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Model summary
    print(f"\nTraining complete!")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Total parameters: {model.count_params():,}")
    
    return model, data_loader

if __name__ == "__main__":
    model, data_loader = main()