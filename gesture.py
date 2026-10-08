#!/usr/bin/env python3
"""Simple hand gesture tracking using MediaPipe Hands."""

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Hand landmark connections for drawing
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

def get_fingertip_landmarks():
    """Return landmark indices for fingertips."""
    return [
        mp_hands.HandLandmark.THUMB_TIP,      # Thumb tip (4)
        mp_hands.HandLandmark.INDEX_FINGER_TIP,  # Index (8)
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP, # Middle (12)
        mp_hands.HandLandmark.RING_FINGER_TIP,   # Ring (16)
        mp_hands.HandLandmark.PINKY_TIP        # Pinky (20)
    ]

def detect_gesture(landmarks, handedness):
    """Simple gesture detection based on finger positions."""
    tips = get_fingertip_landmarks()

    # Check which fingers are extended (tip is above PIP joint)
    fingers_extended = []

    # Thumb (compare x position for left/right hand)
    thumb_tip = landmarks[tips[0]]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    if handedness == 'Right':
        fingers_extended.append(thumb_tip.x < thumb_ip.x)
    else:
        fingers_extended.append(thumb_tip.x > thumb_ip.x)

    # Other fingers (compare y position - lower y means higher on screen)
    for i, tip_idx in enumerate(tips[1:], 1):
        pip_idx = tip_idx - 2  # PIP joint is 2 landmarks before tip
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        fingers_extended.append(tip.y < pip.y)

    # Gesture recognition
    if all(fingers_extended):
        return "OPEN_PALM"
    elif not any(fingers_extended):
        return "FIST"
    elif fingers_extended[1] and not fingers_extended[2]:  # Index only
        return "POINTING"
    elif fingers_extended[1] and fingers_extended[2] and not fingers_extended[3]:  # Index + Middle
        return "PEACE"
    elif fingers_extended[0] and fingers_extended[4] and not any(fingers_extended[1:4]):  # Thumb + Pinky
        return "ROCK"

    return "CUSTOM"

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Hand Gesture Tracking Started")
    print("Press 'q' to quit")
    print("-" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw landmarks and detect gestures
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness (Left/Right)
                handedness = "Unknown"
                if results.multi_handedness:
                    handedness = results.multi_handedness[hand_idx].classification[0].label

                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

                # Detect gesture
                gesture = detect_gesture(hand_landmarks.landmark, handedness)

                # Display gesture and handedness
                y_offset = 50 + (hand_idx * 80)
                cv2.putText(frame, f"Hand: {handedness}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Gesture: {gesture}", (10, y_offset + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Print to console
                print(f"Detected: {handedness} hand - {gesture}")

        # Show frame
        cv2.imshow('Hand Gesture Tracking', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Gesture tracking stopped")

if __name__ == "__main__":
    main()
