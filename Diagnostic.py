import os
import json
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("Gesture Model Diagnostics & Improvement Script")
print("="*60)

# Configuration
DATA_DIR = "gesture_training_data"
MODEL_SAVE_PATH = "gesture_model_improved.h5"
SCALER_SAVE_PATH = "gesture_scaler_improved.pkl"
LABEL_ENCODER_SAVE_PATH = "gesture_label_encoder_improved.pkl"

# Load existing components for analysis
def load_existing_model():
    """Load existing model components for analysis"""
    try:
        model = tf.keras.models.load_model("gesture_model.h5")
        with open("gesture_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open("gesture_label_encoder.pkl", 'rb') as f:
            label_encoder = pickle.load(f)
        print("✓ Existing model components loaded successfully")
        return model, scaler, label_encoder
    except Exception as e:
        print(f"✗ Could not load existing model: {e}")
        return None, None, None

# Analyze training data quality
def analyze_training_data():
    """Analyze the quality and distribution of training data"""
    print("\n1. ANALYZING TRAINING DATA QUALITY")
    print("-" * 40)
    
    if not os.path.exists(DATA_DIR):
        print(f"✗ Training data directory '{DATA_DIR}' not found!")
        return {}
    
    gesture_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json') and f != 'summary.json']
    
    if not gesture_files:
        print("✗ No training data files found!")
        return {}
    
    data_analysis = {
        'total_samples': 0,
        'gesture_counts': {},
        'sequence_lengths': [],
        'gesture_durations': {},
        'motion_analysis': {},
        'data_quality_issues': []
    }
    
    print(f"Found {len(gesture_files)} training files")
    
    for filename in gesture_files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            gesture = data['label']
            sequence_length = len(data['sequence'])
            duration = data.get('duration', 0)
            
            # Count samples per gesture
            if gesture not in data_analysis['gesture_counts']:
                data_analysis['gesture_counts'][gesture] = 0
                data_analysis['gesture_durations'][gesture] = []
                data_analysis['motion_analysis'][gesture] = []
            
            data_analysis['gesture_counts'][gesture] += 1
            data_analysis['total_samples'] += 1
            data_analysis['sequence_lengths'].append(sequence_length)
            data_analysis['gesture_durations'][gesture].append(duration)
            
            # Analyze motion in the sequence
            motion_magnitudes = []
            for frame in data['sequence']:
                if 'hands' in frame and frame['hands']:
                    motion = frame['hands'][0].get('motion', {})
                    motion_mag = motion.get('overall_motion_magnitude', 0)
                    motion_magnitudes.append(motion_mag)
            
            avg_motion = np.mean(motion_magnitudes) if motion_magnitudes else 0
            data_analysis['motion_analysis'][gesture].append(avg_motion)
            
            # Check for potential data quality issues
            if sequence_length < 10:
                data_analysis['data_quality_issues'].append(f"{filename}: Very short sequence ({sequence_length} frames)")
            
            if avg_motion < 0.1 and 'wave' in gesture.lower() or 'forward' in gesture.lower():
                data_analysis['data_quality_issues'].append(f"{filename}: Motion gesture with low motion ({avg_motion:.3f})")
                
        except Exception as e:
            print(f"✗ Error reading {filename}: {e}")
            data_analysis['data_quality_issues'].append(f"{filename}: File read error")
    
    # Print analysis results
    print(f"✓ Total samples: {data_analysis['total_samples']}")
    print("\nSamples per gesture:")
    for gesture, count in sorted(data_analysis['gesture_counts'].items()):
        print(f"  {gesture}: {count} samples")
    
    print(f"\nSequence length statistics:")
    lengths = data_analysis['sequence_lengths']
    print(f"  Min: {min(lengths)} frames")
    print(f"  Max: {max(lengths)} frames") 
    print(f"  Average: {np.mean(lengths):.1f} frames")
    print(f"  Median: {np.median(lengths):.1f} frames")
    
    # Check class imbalance
    counts = list(data_analysis['gesture_counts'].values())
    if max(counts) / min(counts) > 5:
        print(f"\n⚠️  SEVERE CLASS IMBALANCE detected! Ratio: {max(counts)/min(counts):.1f}:1")
        data_analysis['data_quality_issues'].append("Severe class imbalance")
    
    # Check minimum samples per class
    min_samples = min(counts)
    if min_samples < 5:
        print(f"⚠️  INSUFFICIENT SAMPLES: Some gestures have only {min_samples} samples")
        data_analysis['data_quality_issues'].append(f"Insufficient samples (min: {min_samples})")
    
    # Motion analysis for motion gestures
    print(f"\nMotion analysis:")
    motion_gestures = ['wave', 'go_forward', 'come_here', 'clockwise', 'counter_clockwise']
    for gesture in motion_gestures:
        if gesture in data_analysis['motion_analysis']:
            motions = data_analysis['motion_analysis'][gesture]
            avg_motion = np.mean(motions)
            print(f"  {gesture}: {avg_motion:.3f} cm/s average motion")
    
    if data_analysis['data_quality_issues']:
        print(f"\n⚠️  DATA QUALITY ISSUES FOUND:")
        for issue in data_analysis['data_quality_issues'][:10]:  # Show first 10
            print(f"  • {issue}")
        if len(data_analysis['data_quality_issues']) > 10:
            print(f"  ... and {len(data_analysis['data_quality_issues']) - 10} more issues")
    
    return data_analysis

def create_improved_model(input_shape, num_classes, data_analysis):
    """Create an improved model based on data analysis"""
    print(f"\n3. CREATING IMPROVED MODEL")
    print("-" * 40)
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    # Adjust model complexity based on data amount
    total_samples = data_analysis.get('total_samples', 100)
    
    if total_samples < 50:
        print("⚠️  Very small dataset - using simple model")
        model_complexity = 'simple'
    elif total_samples < 200:
        print("Using medium complexity model")
        model_complexity = 'medium'
    else:
        print("Using complex model")
        model_complexity = 'complex'
    
    if model_complexity == 'simple':
        # Simple model for small datasets
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
    elif model_complexity == 'medium':
        # Medium complexity model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 5, activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
    else:
        # Complex model (original)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 5, activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    # Compile with appropriate learning rate
    initial_lr = 0.001 if total_samples > 100 else 0.0005
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model created with {model.count_params():,} parameters")
    return model

def provide_recommendations(data_analysis):
    """Provide specific recommendations for improvement"""
    print(f"\n4. RECOMMENDATIONS FOR IMPROVEMENT")
    print("-" * 40)
    
    recommendations = []
    
    # Check data quantity
    total_samples = data_analysis.get('total_samples', 0)
    if total_samples < 100:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Data Collection',
            'issue': f'Insufficient training data ({total_samples} samples)',
            'solution': 'Collect at least 10-15 samples per gesture (150+ total samples)',
            'action': 'Record more gesture samples using the training script'
        })
    
    # Check class balance
    gesture_counts = data_analysis.get('gesture_counts', {})
    if gesture_counts:
        counts = list(gesture_counts.values())
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        
        if imbalance_ratio > 3:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Data Balance',
                'issue': f'Class imbalance (ratio: {imbalance_ratio:.1f}:1)',
                'solution': 'Balance dataset by collecting more samples for underrepresented gestures',
                'action': f'Focus on recording more samples for: {[g for g, c in gesture_counts.items() if c == min(counts)]}'
            })
    
    # Check sequence lengths
    sequence_lengths = data_analysis.get('sequence_lengths', [])
    if sequence_lengths:
        avg_length = np.mean(sequence_lengths)
        if avg_length < 30:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Sequence Length',
                'issue': f'Very short sequences (avg: {avg_length:.1f} frames)',
                'solution': 'Record longer gesture sequences (2-4 seconds)',
                'action': 'Hold gestures longer before pressing spacebar to stop recording'
            })
    
    # Check motion gestures
    motion_analysis = data_analysis.get('motion_analysis', {})
    motion_gestures = ['wave', 'go_forward', 'come_here', 'clockwise', 'counter_clockwise']
    
    for gesture in motion_gestures:
        if gesture in motion_analysis:
            motions = motion_analysis[gesture]
            avg_motion = np.mean(motions) if motions else 0
            if avg_motion < 1.0:  # Very low motion for motion gesture
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Motion Quality',
                    'issue': f'{gesture} has very low motion ({avg_motion:.3f} cm/s)',
                    'solution': 'Record with more pronounced hand movement',
                    'action': f'Re-record {gesture} with exaggerated hand movements'
                })
    
    # Feature engineering recommendations
    recommendations.append({
        'priority': 'MEDIUM',
        'category': 'Feature Engineering',
        'issue': 'Model may be overfitting or underfitting',
        'solution': 'Add data augmentation and feature selection',
        'action': 'Consider temporal augmentation, noise injection, and feature importance analysis'
    })
    
    # Model architecture recommendations
    if total_samples < 50:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Model Architecture',
            'issue': 'Model too complex for small dataset',
            'solution': 'Use simpler model architecture',
            'action': 'Reduce model complexity or collect more data'
        })
    
    # Training strategy recommendations
    recommendations.append({
        'priority': 'MEDIUM',
        'category': 'Training Strategy',
        'issue': 'Potential overfitting or poor generalization',
        'solution': 'Improve training strategy',
        'action': 'Use cross-validation, early stopping, and better regularization'
    })
    
    # Print recommendations by priority
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        priority_recs = [r for r in recommendations if r['priority'] == priority]
        if priority_recs:
            print(f"\n🔴 {priority} PRIORITY:")
            for i, rec in enumerate(priority_recs, 1):
                print(f"  {i}. [{rec['category']}] {rec['issue']}")
                print(f"     → Solution: {rec['solution']}")
                print(f"     → Action: {rec['action']}")
                print()
    
    return recommendations

def create_data_collection_plan(data_analysis):
    """Create a specific data collection plan"""
    print(f"\n5. SPECIFIC DATA COLLECTION PLAN")
    print("-" * 40)
    
    gesture_counts = data_analysis.get('gesture_counts', {})
    
    if not gesture_counts:
        print("No existing data found. Recommended starting collection:")
        gestures = ['number_one', 'number_two', 'number_three', 'number_four', 'number_five',
                   'ok_sign', 'thumbs_up', 'thumbs_down', 'wave', 'stop']
        for gesture in gestures:
            print(f"  {gesture}: Record 10-15 samples")
        return
    
    print("Data collection priorities (aim for 10-15 samples per gesture):")
    
    # Sort by current count (lowest first)
    sorted_gestures = sorted(gesture_counts.items(), key=lambda x: x[1])
    
    for gesture, current_count in sorted_gestures:
        target_count = 12  # Target samples per gesture
        needed = max(0, target_count - current_count)
        
        if needed > 0:
            priority = "HIGH" if needed > 5 else "MEDIUM" if needed > 2 else "LOW"
            print(f"  {gesture}: {current_count} → {target_count} ({needed} more needed) [{priority}]")
        else:
            print(f"  {gesture}: {current_count} ✓ (sufficient)")
    
    print(f"\nEstimated time needed: {sum(max(0, 12 - count) for count in gesture_counts.values()) * 2} minutes")
    print("(Assuming 2 minutes per sample including setup)")

def retrain_with_improvements():
    """Retrain the model with improvements"""
    print(f"\n6. RETRAINING MODEL WITH IMPROVEMENTS")
    print("-" * 40)
    
    # This would integrate with your existing training code
    print("To retrain with improvements:")
    print("1. Collect additional data based on recommendations above")
    print("2. Run the improved training script (this would be integrated)")
    print("3. Use cross-validation for better evaluation")
    print("4. Apply data augmentation techniques")
    
    # Placeholder for actual retraining code
    print("\n[Retraining code would go here - integrating with your existing training pipeline]")

def main():
    """Main diagnostic and improvement function"""
    print("Starting comprehensive gesture model diagnostics...\n")
    
    # 1. Load and analyze existing model
    existing_model, existing_scaler, existing_label_encoder = load_existing_model()
    
    # 2. Analyze training data
    data_analysis = analyze_training_data()
    
    if not data_analysis:
        print("Cannot proceed without training data analysis")
        return
    
    # 3. Provide specific recommendations
    recommendations = provide_recommendations(data_analysis)
    
    # 4. Create data collection plan
    create_data_collection_plan(data_analysis)
    
    # 5. Show next steps
    print(f"\n" + "="*60)
    print("IMMEDIATE NEXT STEPS:")
    print("="*60)
    print("1. Focus on HIGH priority recommendations first")
    print("2. Collect more training data following the plan above")
    print("3. Ensure motion gestures have actual motion")
    print("4. Balance your dataset (similar samples per gesture)")
    print("5. Record longer sequences (2-4 seconds each)")
    print("6. Re-run training after data improvements")
    
    print(f"\nKey Issues Summary:")
    total_samples = data_analysis.get('total_samples', 0)
    if total_samples < 100:
        print(f"• CRITICAL: Only {total_samples} samples - need 150+ for good performance")
    
    gesture_counts = data_analysis.get('gesture_counts', {})
    if gesture_counts:
        counts = list(gesture_counts.values())
        if max(counts) / min(counts) > 3:
            print(f"• MAJOR: Severe class imbalance - some gestures have {min(counts)} samples, others {max(counts)}")
    
    data_quality_issues = data_analysis.get('data_quality_issues', [])
    if data_quality_issues:
        print(f"• MODERATE: {len(data_quality_issues)} data quality issues found")
    
    print(f"\nExpected improvement after fixing issues:")
    if total_samples < 50:
        print("• Accuracy should improve from ~random to 60-80% with sufficient data")
    elif max(counts) / min(counts) > 5:
        print("• Accuracy should improve significantly with balanced classes")
    else:
        print("• Should see moderate improvement with data quality fixes")
    
    return data_analysis, recommendations

if __name__ == "__main__":
    try:
        analysis, recs = main()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to close...")