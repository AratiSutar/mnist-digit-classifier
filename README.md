# MNIST Digit Classifier

A handwritten digit recognition system built with PyTorch that achieves 98.26% accuracy on the MNIST test set. Includes a live drawing app where you draw a digit and the network predicts it in real time.

## Demo
Draw a digit with your mouse → network predicts instantly → shows confidence percentage and top 3 guesses

## Model Architecture
- Input: 784 neurons (28×28 pixels flattened)
- Hidden Layer 1: 256 neurons + BatchNorm + ReLU
- Hidden Layer 2: 128 neurons + BatchNorm + ReLU
- Hidden Layer 3: 64 neurons + BatchNorm + ReLU
- Output: 10 neurons (one per digit 0-9)

## Results
| Model | Optimizer | Epochs | Test Accuracy |
|-------|-----------|--------|---------------|
| Simple (784→128→10) | SGD | 5 | 92.20% |
| Improved (784→256→128→64→10) | Adam | 10 | 98.26% |

## Files
- `MINST.py` — model training and evaluation
- `MINST_LO_PRE.PY` — load saved model and run predictions
- `live_digit_prediction.py` — real time pygame drawing app
- `mnist_model.pth` — trained model weights
