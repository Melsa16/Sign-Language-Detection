import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from model import SignLanguageModel

class RealTimeDetector:
    def __init__(self, model_path='models'):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = SignLanguageModel(model_path)
        self.model.load_model()
        
        # Load label map
        with open(Path(model_path) / 'label_map.json', 'r') as f:
            self.label_map = json.load(f)
            
        # Define key face landmarks for mouth area (around 50 points)
        self.face_indices = list(range(0, 17))  # Jaw line
        self.face_indices.extend(list(range(17, 27)))  # Eyebrows
        self.face_indices.extend(list(range(27, 36)))  # Nose
        self.face_indices.extend(list(range(36, 48)))  # Left eye
        self.face_indices.extend(list(range(48, 60)))  # Right eye
        self.face_indices.extend(list(range(61, 68)))  # Outer lip
        self.face_indices.extend(list(range(68, 75)))  # Inner lip
        
        # Define key hand landmarks (10 points per hand)
        self.hand_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        
        # Define action names mapping
        self.action_names = {
            "0": "hello",
            "1": "goodbye",
            "2": "sign",
            "3": "language",
            "4": "yes",
            "5": "no",
            "6": "good",
            "7": "bad",
            "8": "morning"
        }
        
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
    
    def detect(self):
        cap = cv2.VideoCapture(0)
        sequence = []
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Make detections
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                
                # Draw landmarks
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                
                # Extract keypoints
                keypoints = self.extract_keypoints(results)
                sequence.append(keypoints)
                
                # Keep only last 20 frames
                if len(sequence) > 20:
                    sequence = sequence[-20:]
                
                # Make prediction when we have enough frames
                if len(sequence) == 20:
                    prediction = self.model.predict(np.array([sequence]))
                    predicted_class = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_class]
                    
                    # Get the action name from our mapping
                    action_name = self.action_names.get(str(predicted_class), f"Unknown ({predicted_class})")
                    
                    # Draw prediction on frame with larger text
                    cv2.putText(image, f'Action: {action_name}', (15,30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Confidence: {confidence:.2f}', (15,70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('Sign Language Detection', image)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RealTimeDetector()
    detector.detect() 