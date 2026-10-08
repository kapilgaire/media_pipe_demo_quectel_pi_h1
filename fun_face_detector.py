#!/usr/bin/env python3
"""
Fun Facial Action Detector - 5 Actions!
Shows different images based on facial actions:
1. Both eyes closed -> Shows "eye_closed.jpg"
2. Tongue out -> Shows "tounge_outside.jpg"
3. One eye open, one closed -> Shows "one_eye.jpg"
4. Smiling -> Shows "smile.jpg"
5. Default (eyes open, mouth closed) -> Shows "default.jpg"
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

class FunFaceDetector:
    def __init__(self, eye_closed_image_path, tongue_image_path, default_image_path,
                 one_eye_image_path, smile_image_path):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Load images
        self.eye_closed_img = self.load_image(eye_closed_image_path)
        self.tongue_img = self.load_image(tongue_image_path)
        self.default_img = self.load_image(default_image_path)
        self.one_eye_img = self.load_image(one_eye_image_path)
        self.smile_img = self.load_image(smile_image_path)

        # Thresholds
        self.EAR_THRESHOLD = 0.28  # Eye Aspect Ratio for blink (increased for better sensitivity)
        self.MAR_THRESHOLD = 0.6   # Mouth Aspect Ratio for tongue
        self.SMILE_THRESHOLD = 0.28  # Smile detection threshold (relaxed from 0.25 to 0.28)

        # State
        self.current_action = None
        self.action_stable_count = 0
        self.STABLE_THRESHOLD = 5  # Frames to confirm action

        # Smoothing for eye detection (prevent flickering)
        self.left_eye_history = []
        self.right_eye_history = []
        self.EYE_HISTORY_SIZE = 5  # Number of frames to average

    def load_image(self, path):
        """Load an image from file."""
        if not Path(path).exists():
            print(f"Warning: Image not found at {path}")
            print("Please provide the image file")
            return None

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Could not load image from {path}")
            return None

        print(f"Loaded image: {path} - Size: {img.shape}")
        return img

    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR) for blink detection."""
        points = [landmarks[i] for i in eye_indices]

        # Vertical distances
        v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))

        # Horizontal distance
        h = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_tongue(self, landmarks):
        """Detect if tongue is sticking out."""
        # Mouth landmarks
        upper_lip_top = landmarks[13]    # Upper lip
        lower_lip_bottom = landmarks[14]  # Lower lip (chin area)
        mouth_center = landmarks[13]

        # Calculate mouth height
        mouth_height = np.linalg.norm(
            np.array(upper_lip_top) - np.array(lower_lip_bottom)
        )

        # Calculate mouth width
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_width = np.linalg.norm(
            np.array(mouth_left) - np.array(mouth_right)
        )

        # Tongue out = very open mouth
        mar = mouth_height / mouth_width

        return mar > self.MAR_THRESHOLD, mar

    def detect_smile(self, landmarks):
        """Optimized smile detection - simpler and more reliable."""
        # Mouth corners
        left_corner = landmarks[61]
        right_corner = landmarks[291]

        # Upper and lower lip (outer)
        upper_lip = landmarks[0]
        lower_lip = landmarks[17]

        # Inner lip corners (for mouth opening)
        upper_inner = landmarks[13]
        lower_inner = landmarks[14]

        # Calculate key measurements
        mouth_width = np.linalg.norm(np.array(left_corner) - np.array(right_corner))
        mouth_height = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip))
        inner_opening = np.linalg.norm(np.array(upper_inner) - np.array(lower_inner))

        # 1. Check if mouth is open (showing teeth)
        # Inner opening should be at least 20% of mouth width
        teeth_visible = inner_opening > (mouth_width * 0.20)

        # 2. Check if corners are raised (smile shape)
        # Compare corner height to mouth center
        corner_avg_y = (left_corner[1] + right_corner[1]) / 2
        mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
        corners_up = corner_avg_y < mouth_center_y

        # 3. Check mouth width (should be reasonably wide)
        width_threshold = mouth_height * 1.3
        mouth_wide = mouth_width > width_threshold

        # Smile = open mouth + raised corners + wide mouth
        is_smile = teeth_visible and corners_up and mouth_wide

        return is_smile

    def detect_smile_detailed(self, landmarks):
        """Optimized smile detection with metrics."""
        # Mouth corners
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        upper_lip = landmarks[0]
        lower_lip = landmarks[17]
        upper_inner = landmarks[13]
        lower_inner = landmarks[14]

        # Calculate measurements
        mouth_width = np.linalg.norm(np.array(left_corner) - np.array(right_corner))
        mouth_height = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip))
        inner_opening = np.linalg.norm(np.array(upper_inner) - np.array(lower_inner))

        # Check conditions
        teeth_visible = inner_opening > (mouth_width * 0.20)
        corner_avg_y = (left_corner[1] + right_corner[1]) / 2
        mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
        corners_up = corner_avg_y < mouth_center_y
        mouth_wide = mouth_width > (mouth_height * 1.3)

        is_smile = teeth_visible and corners_up and mouth_wide

        return {
            'is_smile': is_smile,
            'teeth_visible': teeth_visible,
            'corners_raised': corners_up,
            'width_ratio': mouth_width / mouth_height if mouth_height > 0 else 0,
            'inner_opening': inner_opening
        }

    def overlay_image(self, background, overlay, position=None):
        """Overlay an image on the background with transparency support."""
        if overlay is None:
            return background

        bg_h, bg_w = background.shape[:2]
        ov_h, ov_w = overlay.shape[:2]

        # Resize overlay if it's too large
        max_size = min(bg_w, bg_h) // 2
        if ov_w > max_size or ov_h > max_size:
            scale = max_size / max(ov_w, ov_h)
            new_w = int(ov_w * scale)
            new_h = int(ov_h * scale)
            overlay = cv2.resize(overlay, (new_w, new_h))
            ov_h, ov_w = overlay.shape[:2]

        # Calculate position (center-right of screen)
        if position is None:
            x = bg_w - ov_w - 50  # 50px from right edge
            y = (bg_h - ov_h) // 2  # Vertically centered
        else:
            x, y = position

        # Handle transparency
        if overlay.shape[2] == 4:  # Has alpha channel
            overlay_rgb = overlay[:, :, :3]
            overlay_alpha = overlay[:, :, 3] / 255.0
        else:
            overlay_rgb = overlay
            overlay_alpha = np.ones((ov_h, ov_w), dtype=float)

        # Ensure overlay fits within bounds
        if x < 0 or y < 0 or x + ov_w > bg_w or y + ov_h > bg_h:
            return background

        # Blend the images
        for c in range(3):
            background[y:y+ov_h, x:x+ov_w, c] = (
                overlay_alpha * overlay_rgb[:, :, c] +
                (1 - overlay_alpha) * background[y:y+ov_h, x:x+ov_w, c]
            )

        return background

    def process_frame(self, frame):
        """Process a frame and detect facial actions."""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with Face Mesh
        results = self.face_mesh.process(rgb_frame)

        detected_action = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert to pixel coordinates
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    landmarks.append((x, y))

                # 1. Check eye closure
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]

                left_ear = self.calculate_eye_aspect_ratio(landmarks, left_eye_indices)
                right_ear = self.calculate_eye_aspect_ratio(landmarks, right_eye_indices)

                # Add to history for smoothing
                self.left_eye_history.append(left_ear)
                self.right_eye_history.append(right_ear)

                # Keep only recent history
                if len(self.left_eye_history) > self.EYE_HISTORY_SIZE:
                    self.left_eye_history.pop(0)
                if len(self.right_eye_history) > self.EYE_HISTORY_SIZE:
                    self.right_eye_history.pop(0)

                # Use smoothed values (average of recent frames)
                left_ear_smooth = sum(self.left_eye_history) / len(self.left_eye_history)
                right_ear_smooth = sum(self.right_eye_history) / len(self.right_eye_history)

                left_eye_closed = left_ear_smooth < self.EAR_THRESHOLD
                right_eye_closed = right_ear_smooth < self.EAR_THRESHOLD

                # Both eyes closed only if BOTH are below threshold
                both_eyes_closed = left_eye_closed and right_eye_closed

                # One eye open/closed (winking) - only one eye closed
                one_eye_closed = (left_eye_closed and not right_eye_closed) or \
                                (right_eye_closed and not left_eye_closed)

                # 2. Check tongue
                tongue_out, mar = self.detect_tongue(landmarks)

                # 3. Check smile (with detailed metrics)
                smile_metrics = self.detect_smile_detailed(landmarks)
                is_smiling = smile_metrics['is_smile']

                # Determine action (priority order)
                if tongue_out:
                    detected_action = "tongue"
                elif both_eyes_closed:
                    detected_action = "eyes_closed"
                elif one_eye_closed:
                    detected_action = "one_eye"
                elif is_smiling:
                    detected_action = "smile"
                else:
                    detected_action = "default"

                # Draw status on frame
                status_y = 30
                if both_eyes_closed:
                    cv2.putText(frame, "Eyes: BOTH CLOSED", (10, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif one_eye_closed:
                    cv2.putText(frame, "Eyes: ONE CLOSED", (10, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                else:
                    cv2.putText(frame, "Eyes: OPEN", (10, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show EAR values for debugging (smoothed)
                status_y += 25
                cv2.putText(frame, f"L-EAR: {left_ear_smooth:.3f}", (10, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                status_y += 20
                cv2.putText(frame, f"R-EAR: {right_ear_smooth:.3f}", (10, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                status_y += 30
                if tongue_out:
                    cv2.putText(frame, "Tongue: OUT", (10, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Tongue: IN", (10, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                status_y += 30
                if is_smiling:
                    cv2.putText(frame, "Smile: YES", (10, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Smile: NO", (10, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Show smile metrics for debugging
                status_y += 25
                cv2.putText(frame, f"Teeth: {'YES' if smile_metrics['teeth_visible'] else 'NO'}",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                status_y += 20
                cv2.putText(frame, f"Corners: {'UP' if smile_metrics['corners_raised'] else 'DOWN'}",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                status_y += 20
                cv2.putText(frame, f"Width ratio: {smile_metrics['width_ratio']:.2f}",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                break  # Only process first face

        # Smooth action transitions
        if detected_action == self.current_action:
            self.action_stable_count += 1
        else:
            self.current_action = detected_action
            self.action_stable_count = 0

        # Show image based on stable action
        if self.action_stable_count >= self.STABLE_THRESHOLD:
            if self.current_action == "eyes_closed" and self.eye_closed_img is not None:
                frame = self.overlay_image(frame, self.eye_closed_img, None)
                cv2.putText(frame, "EYES CLOSED!", (frame.shape[1]//2 - 100, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 3)
            elif self.current_action == "tongue" and self.tongue_img is not None:
                frame = self.overlay_image(frame, self.tongue_img, None)
                cv2.putText(frame, "TONGUE OUT!", (frame.shape[1]//2 - 100, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 3)
            elif self.current_action == "one_eye" and self.one_eye_img is not None:
                frame = self.overlay_image(frame, self.one_eye_img, None)
                cv2.putText(frame, "ONE EYE OPEN!", (frame.shape[1]//2 - 120, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 3)
            elif self.current_action == "smile" and self.smile_img is not None:
                frame = self.overlay_image(frame, self.smile_img, None)
                cv2.putText(frame, "SMILING!", (frame.shape[1]//2 - 80, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
            elif self.current_action == "default" and self.default_img is not None:
                frame = self.overlay_image(frame, self.default_img, None)

        return frame

def main():
    # Image paths - UPDATE THESE to your actual image files
    EYE_CLOSED_IMAGE = "/home/kapil/Downloads/eye_closed.jpg"
    TONGUE_IMAGE = "/home/kapil/Downloads/tounge_outside.jpg"
    DEFAULT_IMAGE = "/home/kapil/Downloads/default.jpg"  # New: Default state image
    ONE_EYE_IMAGE = "/home/kapil/Downloads/one_eye.jpg"  # New: One eye open image
    SMILE_IMAGE = "/home/kapil/Downloads/smile.jpg"      # New: Smile image

    print("="*50)
    print("Fun Facial Action Detector - 5 Actions!")
    print("="*50)
    print("Actions:")
    print("  1. Both eyes closed -> Shows eye closed image")
    print("  2. Tongue out -> Shows tongue image")
    print("  3. One eye open, one closed -> Shows one eye image")
    print("  4. Smiling -> Shows smile image")
    print("  5. Default (eyes open, mouth closed) -> Shows default image")
    print("\nControls:")
    print("  Press 'q' to quit")
    print("="*50)

    # Initialize detector
    detector = FunFaceDetector(
        EYE_CLOSED_IMAGE,
        TONGUE_IMAGE,
        DEFAULT_IMAGE,
        ONE_EYE_IMAGE,
        SMILE_IMAGE
    )

    # Open camera (try indices 2, 3, 0 for your Logitech C270)
    camera_indices = [2, 3, 0]
    cap = None

    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"\nCamera opened on index {idx}")
            break

    if not cap or not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nStarting detection...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Process frame
        frame = detector.process_frame(frame)

        # Show frame
        cv2.imshow('Fun Face Detector', frame)

        # Handle keys
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.face_mesh.close()
    print("\nDemo stopped")

if __name__ == "__main__":
    main()

