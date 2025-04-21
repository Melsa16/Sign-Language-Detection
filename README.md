# Real-Time Sign Language Detection

A deep learning-based system for real-time sign language detection using computer vision and LSTM networks.

## Features

- Real-time sign language detection using webcam
- Support for multiple sign language gestures
- Face and hand landmark detection using MediaPipe
- LSTM-based sequence classification
- Comprehensive model evaluation metrics

## Project Structure

```
.
├── src/
│   ├── model.py              # LSTM model implementation
│   ├── preprocessing.py      # Data preprocessing utilities
│   ├── real_time_detection.py # Real-time detection implementation
│   └── evaluation.py         # Model evaluation metrics
├── data/                     # Dataset directory
├── models/                   # Saved model files
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone "https://github.com/Melsa16/Sign-Language-Detection.git"
cd sign-language-detection
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python src/model.py
```

2. Run real-time detection:
```bash
python src/real_time_detection.py
```

3. Evaluate model performance:
```bash
python src/evaluation.py
```

## Model Architecture

- Input: Sequence of face and hand landmarks (210 features)
- Architecture:
  - 2 LSTM layers (32 and 64 units)
  - 2 Dense layers (32 and 16 units)
  - Dropout (0.3) for regularization
- Output: Multi-class classification for sign language gestures

## Evaluation Metrics

The model evaluation includes:
- Classification accuracy
- Precision, Recall, and F1-score
- Confusion matrix
- ROC curves with AUC scores
- Real-time performance metrics (FPS)
- Model size analysis

## Requirements

- Python 3.8+
- TensorFlow 2.8.0+
- OpenCV 4.5.5+
- MediaPipe 0.8.9+
- Other dependencies listed in requirements.txt

## License

[MIT LICENSE]

## Acknowledgments

- MediaPipe for pose and hand landmark detection
- TensorFlow for deep learning framework
