import cv2
import mediapipe as mp
import time
import math

print("Starting hand pose estimation with multi-edge palm distance calculation...")

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
    Select the measurement that's closest to its calibrated value (most stable)
    Returns the measurement name, current value, and ratio to calibrated value
    """
    if calibrated_measurements is None:
        # During calibration, return the first measurement
        return 'index_to_pinky', current_measurements['index_to_pinky'], 1.0
    
    best_measurement = None
    best_ratio = None
    smallest_deviation = float('inf')
    
    for measurement_name in current_measurements:
        if measurement_name in calibrated_measurements:
            current_val = current_measurements[measurement_name]
            calibrated_val = calibrated_measurements[measurement_name]
            
            if calibrated_val > 0:  # Avoid division by zero
                ratio = current_val / calibrated_val
                # Calculate deviation from 1.0 (perfect match)
                deviation = abs(ratio - 1.0)
                
                if deviation < smallest_deviation:
                    smallest_deviation = deviation
                    best_measurement = measurement_name
                    best_ratio = ratio
    
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

def calculate_fingertip_3d_position(landmark, hand_distance_cm, frame_width, frame_height, focal_length=500):
    """
    Calculate the 3D position of a fingertip in real-world coordinates
    Returns (x_cm, y_cm, z_cm) where z is distance from camera
    """
    if hand_distance_cm is None:
        return None, None, None
    
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
    
    # Z coordinate is the hand distance plus MediaPipe's relative Z
    real_z_cm = hand_distance_cm + (landmark.z * hand_distance_cm * 0.1)  # Scale MediaPipe Z
    
    return real_x_cm, real_y_cm, real_z_cm

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

# FPS tracking
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0
frame_count = 0

# Calibration tracking
distance_measurements = []
focal_length_estimate = FOCAL_LENGTH_PIXELS

print("Starting main loop...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        frame_count += 1
        
        # Calculate FPS
        fps_frame_count += 1
        if fps_frame_count >= 30:
            fps_end_time = time.time()
            current_fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            fps_frame_count = 0

        try:
            # -------------------------------
            # Hand detection
            # -------------------------------
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            hands_detected = 0
            
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
                    
                    # -------------------------------
                    # NORMAL OPERATION MODE
                    # -------------------------------
                    else:
                        # Calculate distance using best palm measurement
                        hand_distance, best_measurement, best_value, ratio = estimate_distance_from_multi_edge_palm(
                            current_measurements, calibrated_palm_measurements, frame_width, focal_length_estimate)
                        
                        if hand_distance is not None:
                            # Draw the best measurement line (highlighted)
                            if best_measurement in measurement_positions:
                                start_pos, end_pos = measurement_positions[best_measurement]
                                cv2.line(frame, (int(start_pos[0]), int(start_pos[1])), 
                                        (int(end_pos[0]), int(end_pos[1])), (0, 255, 0), 3)  # Green for best
                            
                            # Draw other measurements (dimmed)
                            colors = [(100, 100, 255), (100, 255, 255), (255, 100, 255)]  # Dimmed colors
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
                            
                            # Show which measurement is being used
                            best_measurement_display = best_measurement.replace('_', ' ').title()
                            measurement_info = f"Using: {best_measurement_display} ({ratio*100:.0f}% of cal)"
                            cv2.putText(frame, measurement_info, (10, 175 + hand_idx * 120), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            # Show all current measurements
                            y_offset = 195 + hand_idx * 120
                            for i, (measurement_name, value) in enumerate(current_measurements.items()):
                                display_name = measurement_name.replace('_', ' ').title()
                                calibrated_value = calibrated_palm_measurements.get(measurement_name, 0)
                                current_ratio = (value / calibrated_value * 100) if calibrated_value > 0 else 0
                                
                                # Highlight the best measurement
                                color = (0, 255, 0) if measurement_name == best_measurement else (150, 150, 150)
                                cv2.putText(frame, f"{display_name}: {current_ratio:.0f}%", 
                                           (10, y_offset + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            
                            angle_info = f"Angle: {palm_angle:.1f}°"
                            cv2.putText(frame, angle_info, (10, y_offset + len(current_measurements) * 15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                            
                            # Process each fingertip
                            for tip_idx in FINGER_TIPS:
                                lm = hand_landmarks.landmark[tip_idx]
                                x_px = int(lm.x * frame_width)
                                y_px = int(lm.y * frame_height)
                                
                                # Calculate 3D position of fingertip
                                real_x, real_y, real_z = calculate_fingertip_3d_position(
                                    lm, hand_distance, frame_width, frame_height, focal_length_estimate)
                                
                                if real_z is not None:
                                    # Check thresholds
                                    below_y_threshold = y_px >= Y_THRESHOLD
                                    close_to_camera = real_z < DISTANCE_THRESHOLD
                                    
                                    if below_y_threshold or close_to_camera:
                                        status_msg = f"DETECTED! {hand_label} {FINGER_NAMES[tip_idx]} at screen({x_px}, {y_px})"
                                        status_msg += f" world({real_x:.1f}, {real_y:.1f}, {real_z:.1f}cm)"
                                        status_msg += f" [using_{best_measurement}]"
                                        
                                        if below_y_threshold:
                                            status_msg += " [Y-THRESHOLD]"
                                        if close_to_camera:
                                            status_msg += f" [CLOSE: {real_z:.1f}cm < {DISTANCE_THRESHOLD}cm]"
                                        
                                        print(status_msg)

                                    # Color coding
                                    if close_to_camera and below_y_threshold:
                                        color = (0, 0, 255)  # Red
                                    elif close_to_camera:
                                        color = (0, 165, 255)  # Orange
                                    elif below_y_threshold:
                                        color = (0, 255, 255)  # Yellow
                                    else:
                                        color = (0, 255, 0)  # Green
                                    
                                    cv2.circle(frame, (x_px, y_px), 6, color, -1)
                                    
                                    # Show distance on screen
                                    distance_label = f"{real_z:.1f}cm"
                                    cv2.putText(frame, distance_label, (x_px + 8, y_px + 15), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")

        # Draw UI elements
        if not calibration_mode:
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
            cv2.putText(frame, "Multi-Edge Palm Tracking", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Green=Best measurement, Dimmed=Others", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if not calibration_mode:
            cv2.putText(frame, "Green=Normal, Yellow=Y-thresh, Orange=Close, Red=Both", 
                       (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display frame
        cv2.imshow("Multi-Edge Palm Distance Estimation", frame)

        # Check for quit and controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('r'):  # 'r' for restart calibration
            print("\nRestarting calibration...")
            calibration_mode = True
            calibration_frame_count = 0
            calibration_data = []
            calibrated_palm_measurements = None
        elif key == ord('s') and calibration_mode:  # 's' to skip calibration
            print("\nSkipping calibration - using default values")
            calibrated_palm_measurements = {
                'index_to_pinky': 80,
                'thumb_to_pinky': 120,
                'wrist_to_mcp': 90
            }
            calibration_mode = False

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
    print("Done!")

print("Program ended.")
input("Press Enter to close console...")