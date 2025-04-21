import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_path='data'):
        self.data_path = Path(data_path)
        self.actions = None
        
    def load_data(self):
        """Load and preprocess the collected data"""
        self.actions = [action.name for action in self.data_path.iterdir() if action.is_dir()]
        sequences = []
        labels = []
        
        for action_idx, action in enumerate(self.actions):
            action_path = self.data_path / action
            for sequence in range(30):  # 30 sequences per action
                sequence_data = []
                missing_frames = False
                
                # Check if all frames exist for this sequence
                for frame_num in range(20):  # 20 frames per sequence (0 to 19)
                    frame_path = action_path / f"{sequence}_{frame_num}.npy"
                    if not frame_path.exists():
                        missing_frames = True
                        break
                    sequence_data.append(np.load(frame_path))
                
                # Only add complete sequences
                if not missing_frames:
                    sequences.append(sequence_data)
                    labels.append(action_idx)
                    
        return np.array(sequences), np.array(labels)
    
    def prepare_data(self):
        """Prepare data for training"""
        X, y = self.load_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Further split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'num_classes': len(self.actions),
            'sequence_length': 20,
            'num_features': X.shape[2]
        }
    
    def get_label_map(self):
        """Get the mapping between actions and their numerical labels"""
        if self.actions is None:
            self.load_data()  # Load data if not already loaded
        return {num: label for num, label in enumerate(self.actions)}

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = preprocessor.prepare_data()
    print(f"Data shapes:")
    print(f"X_train: {data['X_train'].shape}")
    print(f"X_val: {data['X_val'].shape}")
    print(f"X_test: {data['X_test'].shape}")
    print(f"Number of classes: {data['num_classes']}")
    print(f"Sequence length: {data['sequence_length']}")
    print(f"Number of features: {data['num_features']}")
    print(f"Label map: {preprocessor.get_label_map()}") 