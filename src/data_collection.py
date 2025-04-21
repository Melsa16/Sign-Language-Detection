import cv2
import numpy as np
import os
import mediapipe as mp
from pathlib import Path

class KeypointCollector:
    def __init__(self, data_path='data'):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Define key face landmarks for mouth area (around 50 points)
        self.face_indices = list(range(0, 17))  # Jaw line
        self.face_indices.extend(list(range(17, 27)))  # Eyebrows
        self.face_indices.extend(list(range(27, 36)))  # Nose
        self.face_indices.extend(list(range(36, 48)))  # Left eye
        self.face_indices.extend(list(range(48, 60)))  # Right eye
        self.face_indices.extend(list(range(61, 68)))  # Outer lip
        self.face_indices.extend(list(range(68, 75)))  # Inner lip
        
        # Define key hand landmarks (10 points per hand)
        # 0: wrist, 4: thumb tip, 8: index tip, 12: middle tip, 16: ring tip, 20: pinky tip
        # 2: thumb middle, 6: index middle, 10: middle middle, 14: ring middle, 18: pinky middle
        self.hand_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        
    def extract_keypoints(self, results):
        """Extract optimized keypoints for sign language detection"""
        # Face landmarks (reduced to ~50 points)
        if results.face_landmarks:
            face = np.array([[results.face_landmarks.landmark[i].x, 
                             results.face_landmarks.landmark[i].y, 
                             results.face_landmarks.landmark[i].z] for i in self.face_indices]).flatten()
        else:
            face = np.zeros(len(self.face_indices) * 3)
        
        # Hand landmarks (reduced to 10 points per hand)
        if results.left_hand_landmarks:
            lh = np.array([[results.left_hand_landmarks.landmark[i].x, 
                           results.left_hand_landmarks.landmark[i].y, 
                           results.left_hand_landmarks.landmark[i].z] for i in self.hand_indices]).flatten()
        else:
            lh = np.zeros(len(self.hand_indices) * 3)
            
        if results.right_hand_landmarks:
            rh = np.array([[results.right_hand_landmarks.landmark[i].x, 
                           results.right_hand_landmarks.landmark[i].y, 
                           results.right_hand_landmarks.landmark[i].z] for i in self.hand_indices]).flatten()
        else:
            rh = np.zeros(len(self.hand_indices) * 3)
        
        return np.concatenate([face, lh, rh])

    def collect_data(self, action, num_sequences=30, sequence_length=20):
        cap = cv2.VideoCapture(0)
        
        # Create directory for the action
        action_path = self.data_path / action
        action_path.mkdir(exist_ok=True)
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for sequence in range(num_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Make detections
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    
                    # Draw only hand and face landmarks
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)
                    self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                    self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                    
                    # Extract keypoints
                    keypoints = self.extract_keypoints(results)
                    
                    # Save keypoints
                    npy_path = action_path / f"{sequence}_{frame_num}.npy"
                    np.save(npy_path, keypoints)
                    
                    # Show frame
                    cv2.putText(image, f'Collecting frames for {action} Video {sequence} Frame {frame_num}', 
                              (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # List of actions to collect
    ACTIONS = [
        "hello",
        "goodbye",
        "sign",
        "language",
        "yes",
        "no",
        "good",
        "bad",
        "morning"
    ]
    
    collector = KeypointCollector()
    
    print("Starting data collection for the following signs:")
    for i, action in enumerate(ACTIONS, 1):
        print(f"{i}. {action}")
    
    print("\nInstructions:")
    print("1. Position yourself in front of the camera")
    print("2. Make sure your face and hands are visible")
    print("3. Perform each sign clearly and consistently")
    print("4. Hold each sign for about 2-3 seconds")
    print("5. Press 'q' to skip to the next sign")
    print("6. Press 'ESC' to exit the program")
    
    input("\nPress Enter to start collecting data...")
    
    for action in ACTIONS:
        print(f"\nCollecting data for sign: {action}")
        print("Press 'q' to skip to next sign or 'ESC' to exit")
        collector.collect_data(action, num_sequences=30, sequence_length=20)
        
    print("\nData collection completed!") 