Handwriting Digit Recognition with PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The model is trained on grayscale images of digits (0-9) and achieves predictions using a softmax output layer.


Features:

1. Dataset Handling:
- Utilizes the MNIST dataset with torchvision.
- Automatically downloads the dataset if not available locally.
- Applies the ToTensor transform for preprocessing.

2. Model Architecture:
- A CNN with two convolutional layers, dropout for regularization, and two fully connected layers.
- Outputs probabilities for each digit class (0-9) using a softmax activation.

3. Training and Testing:
- Custom training and testing functions with batch processing.
- Calculates training loss and evaluates accuracy on the test dataset.

4. Device Compatibility:
- Automatically switches between CPU and GPU based on availability.

5. Visualization:
- Visualizes individual test samples and their corresponding predictions using Matplotlib.


Results:
- The model achieves over 95% accuracy on the test dataset after training for 10 epochs.
- Visualization allows checking predictions for individual test samples.
