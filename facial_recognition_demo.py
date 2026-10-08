#!/usr/bin/env python3
"""
Advanced Facial Recognition Demo
Features: Face detection, smile detection, eye blink detection,
          mouth opening, head pose estimation, and facial landmarks
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class FaceAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Drawing specs
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1,
            circle_radius=1,
            color=(0, 255, 0)
        )

        # State tracking
        self.blink_counter = 0
        self.blink_history = deque(maxlen=30)
        self.mouth_history = deque(maxlen=10)
        self.smile_history = deque(maxlen=10)
        self.head_pose_history = deque(maxlen=5)

        # Thresholds
        self.EAR_THRESHOLD = 0.25  # Eye Aspect Ratio for blink
        self.MAR_THRESHOLD = 0.5   # Mouth Aspect Ratio
        self.SMILE_THRESHOLD = 0.3

    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR) for blink detection."""
        # Extract the 6 key eye points
        points = [landmarks[i] for i in eye_indices]

        # Vertical distances
        v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))

        # Horizontal distance
        h = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

        # EAR calculation
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def calculate_mouth_aspect_ratio(self, landmarks):
        """Calculate Mouth Aspect Ratio (MAR) for mouth opening."""
        # Mouth landmarks
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]

        # Vertical distance
        v = np.linalg.norm(np.array(mouth_top) - np.array(mouth_bottom))

        # Horizontal distance
        h = np.linalg.norm(np.array(mouth_left) - np.array(mouth_right))

        mar = v / h
        return mar

    def detect_smile(self, landmarks):
        """Detect smile based on mouth corner positions."""
        # Mouth corners
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        mouth_center = landmarks[13]

        # Calculate mouth width
        mouth_width = np.linalg.norm(np.array(left_corner) - np.array(right_corner))

        # Calculate mouth height
        mouth_height = np.linalg.norm(
            np.array(mouth_center) -
            np.array([(left_corner[0] + right_corner[0]) / 2,
                     (left_corner[1] + right_corner[1]) / 2])
        )

        # Smile ratio
        smile_ratio = mouth_height / mouth_width

        return smile_ratio < self.SMILE_THRESHOLD

    def estimate_head_pose(self, landmarks, frame_shape):
        """Estimate head pose using facial landmarks."""
        # More 3D model points for better accuracy
        # Nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
        image_points = np.array([
            landmarks[1],      # Nose tip
            landmarks[152],    # Chin
            landmarks[61],     # Left eye left corner
            landmarks[291],    # Right eye right corner
            landmarks[61],     # Left mouth corner
            landmarks[291],    # Right mouth corner
        ], dtype="double")

        # 3D model points (approximate facial model)
        model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -330.0, -65.0),    # Chin
            (-165.0, 170.0, -135.0), # Left eye left corner
            (165.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),# Left mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Camera matrix (approximate)
        focal_length = frame_shape[1]
        center = (frame_shape[1] / 2, frame_shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        # Distortion coefficients
        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        if success:
            # Convert to Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

            return {
                'yaw': angles[1],      # Left/Right
                'pitch': angles[0],    # Up/Down
                'roll': angles[2]      # Tilt
            }

        return None

    def analyze_frame(self, frame):
        """Analyze a single frame for all facial features."""
        results = {}

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with Face Mesh
        face_results = self.face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Convert landmarks to pixel coordinates
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    landmarks.append((x, y))

                # 1. Eye blink detection
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]

                left_ear = self.calculate_eye_aspect_ratio(landmarks, left_eye_indices)
                right_ear = self.calculate_eye_aspect_ratio(landmarks, right_eye_indices)
                avg_ear = (left_ear + right_ear) / 2.0

                self.blink_history.append(avg_ear)
                eyes_closed = avg_ear < self.EAR_THRESHOLD

                if eyes_closed:
                    self.blink_counter += 1

                # 2. Mouth detection
                mar = self.calculate_mouth_aspect_ratio(landmarks)
                self.mouth_history.append(mar)
                mouth_open = mar > self.MAR_THRESHOLD

                # 3. Smile detection
                is_smiling = self.detect_smile(landmarks)
                self.smile_history.append(1 if is_smiling else 0)
                smile_confidence = sum(self.smile_history) / len(self.smile_history)

                # 4. Head pose
                head_pose = self.estimate_head_pose(landmarks, frame.shape)
                if head_pose:
                    self.head_pose_history.append(head_pose)

                results = {
                    'landmarks': landmarks,
                    'face_landmarks': face_landmarks,
                    'blink_counter': self.blink_counter,
                    'eyes_closed': eyes_closed,
                    'ear': avg_ear,
                    'mouth_open': mouth_open,
                    'mar': mar,
                    'is_smiling': is_smiling,
                    'smile_confidence': smile_confidence,
                    'head_pose': head_pose
                }

                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(200, 200, 200),
                        thickness=1
                    )
                )

                break  # Only process first face

        return results

def main():
    # Initialize
    analyzer = FaceAnalyzer()

    # Open camera (change index if needed: 0, 2, or 3)
    camera_indices = [2, 3, 0]  # Try these in order
    cap = None

    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened on index {idx}")
            break

    if not cap or not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "="*50)
    print("Advanced Facial Recognition Demo")
    print("="*50)
    print("Features:")
    print("  - Eye blink detection")
    print("  - Smile detection")
    print("  - Mouth opening detection")
    print("  - Head pose estimation (yaw/pitch/roll)")
    print("  - Real-time facial landmarks")
    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Press 'r' to reset counters")
    print("="*50 + "\n")

    frame_count = 0
    fps_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Analyze frame
        results = analyzer.analyze_frame(frame)

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()

        # Display overlay
        y_offset = 30

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

        if results:
            # Eye status
            eye_status = "CLOSED" if results['eyes_closed'] else "OPEN"
            eye_color = (0, 0, 255) if results['eyes_closed'] else (0, 255, 0)
            cv2.putText(frame, f"Eyes: {eye_status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
            y_offset += 30

            # Blink counter
            cv2.putText(frame, f"Blinks: {results['blink_counter']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30

            # Mouth status
            mouth_status = "OPEN" if results['mouth_open'] else "CLOSED"
            mouth_color = (0, 0, 255) if results['mouth_open'] else (0, 255, 0)
            cv2.putText(frame, f"Mouth: {mouth_status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)
            y_offset += 30

            # Smile detection
            smile_status = "YES" if results['is_smiling'] else "NO"
            smile_color = (0, 255, 0) if results['is_smiling'] else (0, 0, 255)
            cv2.putText(frame, f"Smiling: {smile_status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, smile_color, 2)
            y_offset += 30

            # Smile confidence
            cv2.putText(frame, f"Smile Confidence: {results['smile_confidence']*100:.1f}%",
                       (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30

            # Head pose
            if results['head_pose']:
                pose = results['head_pose']
                cv2.putText(frame, f"Head Yaw: {pose['yaw']:.1f}°", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
                cv2.putText(frame, f"Head Pitch: {pose['pitch']:.1f}°", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30

                # Head direction alerts
                if abs(pose['yaw']) > 30:
                    direction = "LEFT" if pose['yaw'] > 0 else "RIGHT"
                    cv2.putText(frame, f"Looking {direction}!", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    y_offset += 30

                if abs(pose['pitch']) > 20:
                    direction = "UP" if pose['pitch'] < 0 else "DOWN"
                    cv2.putText(frame, f"Looking {direction}!", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Print to console periodically
            if frame_count % 60 == 0:
                print(f"Blinks: {results['blink_counter']}, "
                      f"Smile: {'Yes' if results['is_smiling'] else 'No'}, "
                      f"Mouth: {'Open' if results['mouth_open'] else 'Closed'}, "
                      f"Eyes: {'Closed' if results['eyes_closed'] else 'Open'}")

        # Show frame
        cv2.imshow('Advanced Facial Recognition', frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            analyzer.blink_counter = 0
            print("Counters reset!")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    analyzer.face_mesh.close()
    print("\nFacial recognition demo stopped")

if __name__ == "__main__":
    main()

