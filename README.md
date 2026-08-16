
# CNN Model for CIFAR-10 Subset Classification

This repository contains the code and results for a Convolutional Neural Network (CNN) model trained on a subset of the CIFAR-10 dataset (automobile, bird, cat, dog) and tested on custom smartphone photos.


#### Model Architecture

The CNN model consists of three convolutional blocks, followed by three fully connected (linear) layers.

**Feature Map Layers:**
-   **Block 1:**
    -   `nn.Conv2d(3, 64, 3, padding=1)`: Input 3 channels (RGB image), 64 output channels, 3x3 kernel, 1 pixel padding.
    -   `nn.BatchNorm2d(64)`
    -   `F.max_pool2d(x, 2)`: Max pooling with 2x2 kernel.
    -   `F.leaky_relu(x)`: Leaky ReLU activation.
    -   Output size: 16x16 feature maps.

-   **Block 2:**
    -   `nn.Conv2d(64, 128, 3)`: Input 64 channels, 128 output channels, 3x3 kernel (no padding).
    -   `nn.BatchNorm2d(128)`
    -   `F.max_pool2d(x, 2)`: Max pooling with 2x2 kernel.
    -   `F.leaky_relu(x)`: Leaky ReLU activation.
    -   Output size: 7x7 feature maps.

-   **Block 3:**
    -   `nn.Conv2d(128, 256, 3)`: Input 128 channels, 256 output channels, 3x3 kernel (no padding).
    -   `nn.BatchNorm2d(256)`
    -   `F.max_pool2d(x, 2)`: Max pooling with 2x2 kernel.
    -   `F.leaky_relu(x)`: Leaky ReLU activation.
    -   Output size: 2x2 feature maps.

**Linear Decision Layers:**
-   The output from the last convolutional block is flattened to a vector.
-   `nn.Linear(2*2*256, 256)`: First fully connected layer.
-   `F.dropout(x, p=0.5, training=self.training)`: Dropout layer for regularization.
-   `nn.Linear(256, 64)`: Second fully connected layer.
-   `F.dropout(x, p=0.5, training=self.training)`: Dropout layer.
-   `nn.Linear(64, num_classes)`: Output layer, `num_classes` is 4 for the target classes.

The network uses `Leaky ReLU` activation functions in the hidden layers and `CrossEntropyLoss` for the loss function, optimized with `Adam`.


## Training Results

The model was trained for 10 epochs. Below are the final performance metrics:

-   **Training Loss (CEL):** 0.149
-   **Development (Validation) Loss (CEL):** 0.679
-   **Test Loss (CEL):** 0.606
-   **Training Accuracy:** 94.95%
-   **Development (Validation) Accuracy:** 82.69%
-   **Test Accuracy:** 82.35%

### Training History

![Training History](model/training_history.png)

### Confusion Matrix (Test Set)

![Confusion Matrix](model/confusion_matrix.png)

## Real-world Testing

The model was also evaluated on a set of custom smartphone photos. The predictions for these images are displayed in the notebook.
