# CNN for MNIST Handwritten Digit Classification

## Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset with high accuracy. The model is designed with multiple convolutional layers and optimized using the Adam optimizer for efficient training and fast convergence.

## Features
- Loads the MNIST dataset and preprocesses images.
- Implements a CNN with two convolutional layers and three fully connected layers.
- Uses the Adam optimizer and cross-entropy loss for training.
- Evaluates model accuracy on the test set.
- Visualizes training loss over epochs.
- Saves the trained model for future inference.

## Dependencies
Ensure you have the following libraries installed:

```bash
pip install torch torchvision matplotlib numpy
```

## File Structure
- `mnist_cnn.py`: Main script containing model definition, training, testing, and visualization functions.
- `mnist_cnn_model.pth`: Saved model after training.
- `README.md`: Project documentation.

## Model Architecture
The CNN consists of:
- Two convolutional layers with ReLU activation.
- Fully connected layers for classification.
- Softmax activation at the output layer for digit classification.

## Usage
### 1. Run the model
To train and evaluate the model, execute the following command:

```bash
python mnist_cnn.py
```

### 2. Train the model
The script will automatically train the model for 15 epochs with a learning rate of 0.0005. Training progress will be displayed in the console.

### 3. Evaluate the model
After training, the model will be tested on the MNIST test set, and the accuracy will be displayed.

### 4. View Training Loss Curve
A plot showing the training loss over epochs will be displayed after training is complete.

### 5. Save the Model
The trained model will be saved as `mnist_cnn_model.pth` for future use.

## Results
- The model achieves high accuracy on the MNIST test set.
- Training loss decreases steadily, ensuring stable convergence.
- The Adam optimizer helps improve learning efficiency.

## Future Improvements
- Implement dropout layers to reduce overfitting.
- Experiment with deeper architectures and different optimizers.
- Fine-tune hyperparameters for improved performance.

## Author
Vivaan Chugh

## License
This project is open-source and free to use for educational purposes.

