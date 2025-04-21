import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from pathlib import Path
import json

class SignLanguageModel:
    def __init__(self, model_path='models'):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model = None
        
    def build_model(self, num_classes, sequence_length, num_features):
        """Build optimized LSTM model for sign language detection"""
        # Calculate approximate feature dimensions
        # ~50 face points * 3 coordinates + 10 hand points * 3 coordinates * 2 hands
        # = 150 + 60 = 210 features
        
        model = Sequential([
            # First LSTM layer with reduced units
            LSTM(32, return_sequences=True, activation='relu', input_shape=(sequence_length, num_features)),
            
            # Second LSTM layer with reduced units
            LSTM(64, return_sequences=False, activation='relu'),
            
            # Dense layers with reduced units
            Dense(32, activation='relu'),
            Dropout(0.3),  # Reduced dropout
            Dense(16, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model = model
        return model
    
    def train(self, data, epochs=100, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model(
                num_classes=data['num_classes'],
                sequence_length=data['sequence_length'],
                num_features=data['num_features']
            )
        
        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(data['y_train'])
        y_val = tf.keras.utils.to_categorical(data['y_val'])
        
        # Setup callbacks
        checkpoint = ModelCheckpoint(
            str(self.model_path / 'best_model.h5'),  # Convert Path to string
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            data['X_train'],
            y_train,
            validation_data=(data['X_val'], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping]
        )
        
        # Save label map
        label_map = {str(k): str(v) for k, v in enumerate(range(data['num_classes']))}
        with open(self.model_path / 'label_map.json', 'w') as f:
            json.dump(label_map, f)
            
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model has not been built or loaded")
            
        y_test = tf.keras.utils.to_categorical(y_test)
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, sequence):
        """Make prediction on a single sequence"""
        if self.model is None:
            raise ValueError("Model has not been built or loaded")
            
        # Reshape sequence if needed
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)
            
        return self.model.predict(sequence)
    
    def save_model(self):
        """Save the model"""
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(str(self.model_path / 'final_model.h5'))  # Convert Path to string
        
    def load_model(self, model_name='best_model.h5'):
        """Load a saved model"""
        model_file = self.model_path / model_name
        if not model_file.exists():
            raise FileNotFoundError(f"Model file {model_file} not found")
            
        self.model = tf.keras.models.load_model(str(model_file))  # Convert Path to string
        
        # Load label map
        with open(self.model_path / 'label_map.json', 'r') as f:
            self.label_map = json.load(f)

if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    
    # Example usage
    preprocessor = DataPreprocessor()
    data = preprocessor.prepare_data()
    
    model = SignLanguageModel()
    history = model.train(data, epochs=50)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(data['X_test'], data['y_test'])
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save_model() 