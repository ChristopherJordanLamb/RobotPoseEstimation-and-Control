import cv2
import mediapipe as mp
import time
import math
import torch
import numpy as np
from torchvision.transforms import Compose
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

print("Starting hand pose estimation with multi-edge palm distance calculation and MiDaS depth integration...")

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
# Load MiDaS small model for depth
# -------------------------------
print("Loading MiDaS small model...")
midas = MidasNet_small("weights/dpt_small-midas-2f21e586.pt")  # Provide correct path
midas.eval()
transform = Compose([
    Resize(256, 256),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet()
])

# -------------------------------
# Hand size estimation constants
# -------------------------------
AVERAGE_HAND_SPAN = 18.0  # cm
AVERAGE_HAND_LENGTH = 18.5
AVERAGE_PALM_WIDTH = 8.5

CAMERA_FOV_DEGREES = 60
FOCAL_LENGTH_PIXELS = 500

# -------------------------------
# Calibration system
# -------------------------------
CALIBRATION_FRAMES = 60
calibrated_palm_measurements = None
calibration_data = []
calibration_mode = True
calibration_frame_count = 0

print("CALIBRATION MODE ACTIVE")
print(f"Collecting data for {CALIBRATION_FRAMES} frames...")
print("Press 'r' to restart, 's' to skip")

# -------------------------------
# Multi-edge palm measurement functions
# -------------------------------
def calculate_all_palm_measurements(landmarks, frame_width, frame_height):
    measurements = {}
    positions = {}
    
    wrist = landmarks[0]
    thumb_cmc = landmarks[1]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]
    
    points = {}
    for name, landmark in [
        ('wrist', wrist), ('thumb_cmc', thumb_cmc), ('index_mcp', index_mcp),
        ('middle_mcp', middle_mcp), ('ring_mcp', ring_mcp), ('pinky_mcp', pinky_mcp)
    ]:
        points[name] = (landmark.x * frame_width, landmark.y * frame_height)
    
    index_to_pinky = math.sqrt(
        (points['index_mcp'][0] - points['pinky_mcp'][0])**2 + 
        (points['index_mcp'][1] - points['pinky_mcp'][1])**2
    )
    measurements['index_to_pinky'] = index_to_pinky
    positions['index_to_pinky'] = (points['index_mcp'], points['pinky_mcp'])
    
    thumb_to_pinky = math.sqrt(
        (points['thumb_cmc'][0] - points['pinky_mcp'][0])**2 + 
        (points['thumb_cmc'][1] - points['pinky_mcp'][1])**2
    )
    measurements['thumb_to_pinky'] = thumb_to_pinky
    positions['thumb_to_pinky'] = (points['thumb_cmc'], points['pinky_mcp'])
    
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
    if calibrated_measurements is None:
        return 'index_to_pinky', current_measurements['index_to_pinky'], 1.0
    
    best_measurement = None
    best_ratio = None
    smallest_deviation = float('inf')
    
    for measurement_name in current_measurements:
        if measurement_name in calibrated_measurements:
            current_val = current_measurements[measurement_name]
            calibrated_val = calibrated_measurements[measurement_name]
            
            if calibrated_val > 0:
                ratio = current_val / calibrated_val
                deviation = abs(ratio - 1.0)
                
                if deviation < smallest_deviation:
                    smallest_deviation = deviation
                    best_measurement = measurement_name
                    best_ratio = ratio
    
    if best_measurement is None:
        return 'index_to_pinky', current_measurements['index_to_pinky'], 1.0
    
    return best_measurement, current_measurements[best_measurement], best_ratio

def calculate_palm_orientation(landmarks, frame_width, frame_height):
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

# -------------------------------
# Configuration
# -------------------------------
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_NAMES = {4:"Thumb", 8:"Index", 12:"Middle", 16:"Ring", 20:"Pinky"}
Y_THRESHOLD = 300
DISTANCE_THRESHOLD = 30.0

# -------------------------------
# Open webcam
# -------------------------------
print("Opening webcam...")
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    input("Press Enter to exit...")
    exit(1)

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
# -------------------------------
# Helper: Enhanced fingertip depth using MiDaS
# -------------------------------
def calculate_enhanced_fingertip_depth(landmarks, fingertip_idx, hand_distance_cm, frame_width, frame_height):
    try:
        # Convert frame to RGB and prepare input for MiDaS
        dummy_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        input_batch = transform(dummy_img).unsqueeze(0)  # [1,3,H,W]
        
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(frame_height, frame_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.numpy()
        fingertip = landmarks[fingertip_idx]
        px, py = int(fingertip.x * frame_width), int(fingertip.y * frame_height)
        px = np.clip(px, 0, frame_width - 1)
        py = np.clip(py, 0, frame_height - 1)
        
        depth_value = depth_map[py, px]
        z_offset_cm = (depth_value - depth_map.mean()) * 50  # scale factor
        real_z_cm = hand_distance_cm + z_offset_cm
        
        depth_debug = {"combined_offset": z_offset_cm, "raw_depth": depth_value}
        return real_z_cm, depth_debug
    except Exception as e:
        return hand_distance_cm, None

def calculate_fingertip_3d_position(landmark, fingertip_idx, landmarks, hand_distance_cm, frame_width, frame_height, focal_length=500):
    if hand_distance_cm is None:
        return None, None, None, None
    
    pixel_x = landmark.x * frame_width
    pixel_y = landmark.y * frame_height
    
    centered_x = pixel_x - (frame_width / 2)
    centered_y = pixel_y - (frame_height / 2)
    
    real_x_cm = (centered_x * hand_distance_cm) / focal_length
    real_y_cm = (centered_y * hand_distance_cm) / focal_length
    
    real_z_cm, depth_debug = calculate_enhanced_fingertip_depth(
        landmarks, fingertip_idx, hand_distance_cm, frame_width, frame_height)
    
    return real_x_cm, real_y_cm, real_z_cm, depth_debug

# -------------------------------
# Main loop
# -------------------------------
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0
frame_count = 0
focal_length_estimate = FOCAL_LENGTH_PIXELS

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_count += 1
        fps_frame_count += 1
        
        if fps_frame_count >= 30:
            fps_end_time = time.time()
            current_fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            fps_frame_count = 0
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            hands_detected = 0
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hands_detected = len(results.multi_hand_landmarks)
                
                for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    hand_label = handedness.classification[0].label
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    current_measurements, measurement_positions = calculate_all_palm_measurements(
                        hand_landmarks.landmark, frame_width, frame_height)
                    
                    palm_angle = calculate_palm_orientation(hand_landmarks.landmark, frame_width, frame_height)
                    
                    # --- Calibration ---
                    if calibration_mode:
                        calibration_frame_count += 1
                        calibration_data.append(current_measurements.copy())
                        progress = calibration_frame_count / CALIBRATION_FRAMES
                        cv2.rectangle(frame, (50, 50), (50 + int(300 * progress), 80), (0, 255, 0), -1)
                        cv2.rectangle(frame, (50, 50), (350, 80), (255, 255, 255), 2)
                        cv2.putText(frame, f"CALIBRATING... {calibration_frame_count}/{CALIBRATION_FRAMES}",
                                    (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(frame, "Rotate hand naturally!", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        y_offset = 170
                        for i, (name, val) in enumerate(current_measurements.items()):
                            cv2.putText(frame, f"{name.title()}: {val:.1f}px", (50, y_offset + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                        cv2.putText(frame, f"Palm angle: {palm_angle:.1f}°", (50, y_offset + len(current_measurements)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                        
                        colors = [(0,255,255),(255,255,0),(255,0,255)]
                        for i, (mname, pos) in enumerate(measurement_positions.items()):
                            color = colors[i % len(colors)]
                            start, end = pos
                            cv2.line(frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color,2)
                        
                        if calibration_frame_count >= CALIBRATION_FRAMES:
                            calibrated_palm_measurements = {}
                            for name in current_measurements.keys():
                                vals = [f[name] for f in calibration_data]
                                calibrated_palm_measurements[name] = sum(vals)/len(vals)
                            calibration_mode = False
                            print("\nCALIBRATION COMPLETE!")
                            for name,val in calibrated_palm_measurements.items():
                                print(f"{name.title()}: {val:.1f}px")
                    
                    # --- Normal operation ---
                    else:
                        hand_distance, best_measurement, best_value, ratio = estimate_distance_from_multi_edge_palm(
                            current_measurements, calibrated_palm_measurements, frame_width, focal_length_estimate)
                        
                        if hand_distance is not None:
                            if best_measurement in measurement_positions:
                                start, end = measurement_positions[best_measurement]
                                cv2.line(frame, (int(start[0]),int(start[1])), (int(end[0]),int(end[1])), (0,255,0),3)
                            
                            colors_dim = [(100,100,255),(100,255,255),(255,100,255)]
                            for i, (mname,pos) in enumerate(measurement_positions.items()):
                                if mname != best_measurement:
                                    start,end = pos
                                    cv2.line(frame,(int(start[0]),int(start[1])),(int(end[0]),int(end[1])),colors_dim[i % 3],1)
                            
                            cv2.putText(frame,f"{hand_label}: {hand_distance:.1f}cm",(10,150+hand_idx*120),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),2)
                            best_display = best_measurement.replace("_"," ").title()
                            cv2.putText(frame,f"Using: {best_display} ({ratio*100:.0f}% of cal)",(10,175+hand_idx*120),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                            
                            y_offset = 195 + hand_idx*120
                            for i,(mname,val) in enumerate(current_measurements.items()):
                                calibrated_val = calibrated_palm_measurements.get(mname,0)
                                current_ratio = (val/calibrated_val*100) if calibrated_val>0 else 0
                                color=(0,255,0) if mname==best_measurement else (150,150,150)
                                cv2.putText(frame,f"{mname.title()}: {current_ratio:.0f}%",(10,y_offset+i*15),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
                            cv2.putText(frame,f"Angle: {palm_angle:.1f}°",(10,y_offset+len(current_measurements)*15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(150,150,150),1)
                            
                            for tip_idx in FINGER_TIPS:
                                lm = hand_landmarks.landmark[tip_idx]
                                x_px = int(lm.x*frame_width)
                                y_px = int(lm.y*frame_height)
                                
                                real_x, real_y, real_z, depth_debug = calculate_fingertip_3d_position(
                                    lm, tip_idx, hand_landmarks.landmark, hand_distance, frame_width, frame_height, focal_length_estimate)
                                
                                if real_z is not None:
                                    below_y = y_px>=Y_THRESHOLD
                                    close_to_camera = real_z < DISTANCE_THRESHOLD
                                    
                                    if below_y or close_to_camera:
                                        status_msg = f"DETECTED! {hand_label} {FINGER_NAMES[tip_idx]} at screen({x_px},{y_px})"
                                        status_msg+=f" world({real_x:.1f},{real_y:.1f},{real_z:.1f}cm) [using_{best_measurement}]"
                                        if below_y: status_msg+=" [Y-THRESHOLD]"
                                        if close_to_camera: status_msg+=f" [CLOSE:{real_z:.1f}cm < {DISTANCE_THRESHOLD}cm]"
                                        print(status_msg)
                                    
                                    if close_to_camera and below_y: color=(0,0,255)
                                    elif close_to_camera: color=(0,165,255)
                                    elif below_y: color=(0,255,255)
                                    else: color=(0,255,0)
                                    
                                    cv2.circle(frame,(x_px,y_px),6,color,-1)
                                    cv2.putText(frame,f"{real_z:.1f}cm",(x_px+8,y_px+15),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
                                    if tip_idx==8 and depth_debug:
                                        cv2.putText(frame,f"Z_off:{depth_debug['combined_offset']:.1f}",(x_px+8,y_px+30),cv2.FONT_HERSHEY_SIMPLEX,0.3,(100,100,255),1)
        
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
        
        # Draw UI
        if not calibration_mode:
            cv2.line(frame,(0,Y_THRESHOLD),(frame_width,Y_THRESHOLD),(0,0,255),2)
            cv2.putText(frame,f"Y Threshold: {Y_THRESHOLD}",(frame_width-200,Y_THRESHOLD-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
            cv2.putText(frame,f"Distance Threshold: {DISTANCE_THRESHOLD}cm",(10,frame_height-60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,100,0),1)
        
        cv2.putText(frame,f"FPS: {current_fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        hand_display_y = 30 if calibration_mode else 60
        cv2.putText(frame,f"Hands: {hands_detected}",(10,hand_display_y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        if not calibration_mode:
            cv2.putText(frame,"Multi-Edge Palm Tracking",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.putText(frame,"Green=Best measurement, Dimmed=Others",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
            cv2.putText(frame,"Green=Normal, Yellow=Y-thresh, Orange=Close, Red=Both",(10,frame_height-20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
        
        cv2.imshow("Multi-Edge Palm Distance Estimation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
        elif key == ord('r'):
            print("\nRestarting calibration...")
            calibration_mode=True
            calibration_frame_count=0
            calibration_data=[]
            calibrated_palm_measurements=None
        elif key == ord('s') and calibration_mode:
            print("\nSkipping calibration - using default values")
            calibrated_palm_measurements={'index_to_pinky':80,'thumb_to_pinky':120,'wrist_to_mcp':90}
            calibration_mode=False

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Done!")

print("Program ended.")
input("Press Enter to close console...")
