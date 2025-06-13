# Welcome to Week-3 - Convolutional Neural Networks

## Task-1: Convolutional Neural Networks using Pytorch
- Implementing a convolutional network using the torch library to discriminate between images of cats and dogs.
- Implementing the CNN with the following layers:
  - Convolutional Layer 1:
    - 16 2D filters of kernel size 3 with stride factor of 1 and padding of 1.
    - ReLU layer 1.
    - Max pooling layer 1: Kernel size 2 and stride 2.
  - Convolutional layer 2:
    - 32 2D filters of kernel size 3 with stride factor of 1 and padding of 1.
    - ReLU layer 2.
    - Max pooling layer 2: Kernel size 2 and stride 2.
  - Flatten into a linear layer with 32 x 64 x 64 input nodes and 64 output nodes.
  - ReLU layer 3.
  - Linear layer mapping down 64 dimensions to 1 and a final sigmoid activation that feeds into a binary cross-entropy loss.
- With torchvision modules, augment the training data with transformations like rotating, cropping, flipping, etc. Re-evaluate performance on the test instances.
- Initial Evaluation gave around 63-64% accuracy, but augmenting the data with some transformations gave an accuracy of ~70%

## Task-2: Training CNNs on CIFAR10 and FashionMNIST
- This task involves training Convolutional Neural Networks on two distinct image datasets:
  - Train a CNN on the CIFAR10 dataset.
  - Train a CNN on the FashionMNIST dataset.
- Here, we tried out various architectures, hyperparameters, loss functions, and optimizers to determine which ones work better than the others.
- The accuracy for the FashionMNIST dataset was around 90%, and the accuracy for the CIFAR10 was ~65% (I tried various CNN architectures but failed).
- The CNN Architecture for FashionMNIST dataset is:
  - Convolutional Layer 1:
    - 6 2D filters of kernel size 5 with stride factor of 1 and padding 0.
    - ReLU Layer 1.
    - Max pooling layer 1: Kernel size 2 and stride 2
  - Convolutional Layer 2:
    - 16 2D filters of kernel size 5 with stride factor of 1 and padding 0.
    - ReLU layer 2.
    - Max pooling layer 2: Kernel size 2 and stride 2.
  - Flatten into a linear layer with 16 x 4 x 4 input nodes and 100 output nodes.
  - ReLU layer 3.
  - Linear layer mapping down 100 dimensions to 25.
  - ReLU layer 4.
  - Linear layer mapping down 25 dimensions to 10.
