# network-for-cifar-10
This demonstrates to achieve 90% accuracy on CIFAR-10 dataset
# Environment
- Tensorflow-gpu=2.0.0
- Keras=2.3.1
- 1 GPU with CUDA Version: 10.1
# Network
Convolution layer is a set of 3 operations: Convolution, Activation &amp; Batch normalization. Contain 8 convolution layers, 4 max Pooling layers.
The dropout layer is kept after the Pooling layer. I add 4 dropout layers with the rate is: 0.2, 0.3, 0.4, 0.5. Dropout used to prevent a model from overfitting.
# Result:
Result: 89.050% in test set
Here is the original code: https://github.com/abhijeet3922/Object-recognition-CIFAR-10
