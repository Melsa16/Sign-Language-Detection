import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time
from model import SignLanguageModel
from preprocessing import DataPreprocessor

class ModelEvaluator:
    def __init__(self, model_path='models'):
        self.model = SignLanguageModel(model_path)
        self.model.load_model()
        self.preprocessor = DataPreprocessor()
        
    def evaluate_performance(self):
        """Comprehensive model evaluation"""
        # Load and prepare data
        data = self.preprocessor.prepare_data()
        
        # Get predictions
        y_pred_proba = self.model.predict(data['X_test'])
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = data['y_test']
        
        # Calculate basic metrics
        print("\n=== Classification Report ===")
        print(classification_report(y_true, y_pred, 
                                 target_names=self.preprocessor.get_label_map().values()))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.preprocessor.get_label_map().values(),
                    yticklabels=self.preprocessor.get_label_map().values())
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # ROC Curves for each class
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(self.preprocessor.get_label_map().values()):
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Class')
        plt.legend(loc="lower right")
        plt.savefig('roc_curves.png')
        plt.close()
        
        # Measure inference speed
        inference_times = []
        for _ in range(100):  # Run 100 inferences
            start_time = time.time()
            self.model.predict(data['X_test'][:1])
            inference_times.append(time.time() - start_time)
        
        avg_inference_time = np.mean(inference_times)
        fps = 1.0 / avg_inference_time
        
        print("\n=== Performance Metrics ===")
        print(f"Average Inference Time: {avg_inference_time*1000:.2f} ms")
        print(f"Frames Per Second (FPS): {fps:.2f}")
        
        # Model size
        import os
        model_size = os.path.getsize('models/best_model.h5') / (1024 * 1024)  # Size in MB
        print(f"Model Size: {model_size:.2f} MB")
        
        return {
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': cm,
            'inference_time': avg_inference_time,
            'fps': fps,
            'model_size': model_size
        }

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_performance() 