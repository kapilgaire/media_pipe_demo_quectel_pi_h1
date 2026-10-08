#!/usr/bin/env python3
"""
Real-time Object Detection using Haar Cascades
Built-in OpenCV models - NO downloads needed!
Detects: faces, eyes, smiles, full body, upper body, front face
"""

import cv2
import numpy as np

class HaarCascadeDetector:
    def __init__(self):
        """Initialize Haar Cascade detectors."""
        print("Loading Haar Cascade classifiers...")

        # Load pre-trained classifiers (come with OpenCV)
        cascades = {
            'face': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'eye': cv2.data.haarcascades + 'haarcascade_eye.xml',
            'smile': cv2.data.haarcascades + 'haarcascade_smile.xml',
            'full_body': cv2.data.haarcascades + 'haarcascade_fullbody.xml',
            'upper_body': cv2.data.haarcascades + 'haarcascade_upperbody.xml',
            'front_face': cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        }

        self.detectors = {}
        for name, path in cascades.items():
            try:
                self.detectors[name] = cv2.CascadeClassifier(path)
                print(f"  ✓ Loaded {name} detector")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")

        # Colors for different objects
        self.colors = {
            'face': (0, 255, 0),      # Green
            'eye': (255, 0, 0),       # Blue
            'smile': (0, 255, 255),   # Yellow
            'full_body': (255, 0, 255), # Magenta
            'upper_body': (0, 165, 255), # Orange
        }

        print(f"\nLoaded {len(self.detectors)} detectors")

    def detect_objects(self, frame, enabled_detectors=None):
        """Detect objects in frame."""
        if enabled_detectors is None:
            enabled_detectors = ['face', 'eye', 'smile']

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = {}

        for detector_name in enabled_detectors:
            if detector_name in self.detectors:
                detector = self.detectors[detector_name]

                # Detection parameters
                if detector_name == 'smile':
                    # Smile needs special parameters
                    detected = detector.detectMultiScale(
                        gray,
                        scaleFactor=1.8,
                        minNeighbors=20,
                        minSize=(25, 25),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                elif detector_name == 'eye':
                    # Eyes need different parameters
                    detected = detector.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
                else:
                    # Default parameters for face/body
                    detected = detector.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                results[detector_name] = detected

        return results

    def draw_detections(self, frame, results):
        """Draw detection boxes on frame."""
        total_objects = 0

        for detector_name, detections in results.items():
            if len(detections) > 0:
                color = self.colors.get(detector_name, (255, 255, 255))

                for (x, y, w, h) in detections:
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                    # Draw label
                    label = f"{detector_name}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x, y - label_size[1] - 10),
                                (x + label_size[0], y), color, -1)
                    cv2.putText(frame, label, (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    total_objects += 1

        return frame, total_objects

def main():
    print("="*50)
    print("Haar Cascade Object Detection")
    print("="*50)
    print("Detects (using built-in OpenCV models):")
    print("  • Faces (frontal)")
    print("  • Eyes")
    print("  • Smiles")
    print("  • Full body")
    print("  • Upper body")
    print("\nNo downloads required!")
    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Press '1' - Detect faces only")
    print("  Press '2' - Detect faces + eyes")
    print("  Press '3' - Detect faces + eyes + smiles")
    print("  Press '4' - Detect full body")
    print("  Press '5' - Detect upper body")
    print("="*50)

    # Initialize detector
    detector = HaarCascadeDetector()

    # Open camera
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Start with face + eye detection
    enabled_detectors = ['face', 'eye']

    print("\nStarting detection...\n")

    frame_count = 0
    fps_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Detect objects
        results = detector.detect_objects(frame, enabled_detectors)
        frame, total_objects = detector.draw_detections(frame, results)

        # FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30.0 / ((cv2.getTickCount() - fps_time) / cv2.getTickFrequency())
            fps_time = cv2.getTickCount()

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Objects: {total_objects}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show current mode
            mode_text = f"Mode: {', '.join(enabled_detectors)}"
            cv2.putText(frame, mode_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show frame
        cv2.imshow('Object Detection', frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('1'):
            enabled_detectors = ['face']
            print("Mode: Face detection only")
        elif key == ord('2'):
            enabled_detectors = ['face', 'eye']
            print("Mode: Face + Eye detection")
        elif key == ord('3'):
            enabled_detectors = ['face', 'eye', 'smile']
            print("Mode: Face + Eye + Smile detection")
        elif key == ord('4'):
            enabled_detectors = ['full_body']
            print("Mode: Full body detection")
        elif key == ord('5'):
            enabled_detectors = ['upper_body']
            print("Mode: Upper body detection")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDetection stopped")

if __name__ == "__main__":
    main()

