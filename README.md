# Machine-Learning-Algorithms-Extensions
This is the final project for the CS-GY-6923 Machine Learning course at NYU Tandon School of Engineering.


## Introduction
The goal of this project is to select three basic machine learning algorithms covered in this course, and explore how they could be improved. We were expected to find an extension of the basic algorithm that is fully implemented by a machine learning library such as scikit-learn or Keras, and then implement this extension from scratch using only NumPy. Each extension is implemented on both the [MNIST](http://yann.lecun.com/exdb/mnist/) and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) datasets.

The extensions explored in this project (with links to corresponding notebooks) are:
1. Neural network with softmax activation function in the output layer ([Keras](https://github.com/jmg764/Machine-Learning-Algorithms-Extensions/blob/master/NN%20Softmax%20with%20Keras.ipynb), [NumPy](https://github.com/jmg764/Machine-Learning-Algorithms-Extensions/blob/master/NN%20Softmax%20with%20NumPy.ipynb))
2. Neural network with dropout regularization ([Keras](https://github.com/jmg764/Machine-Learning-Algorithms-Extensions/blob/master/NN%20Dropout%20with%20Keras.ipynb), [NumPy](https://github.com/jmg764/Machine-Learning-Algorithms-Extensions/blob/master/NN%20Dropout%20with%20NumPy.ipynb))
3. Support vector machine with soft margin ([scikit-learn](https://github.com/jmg764/Machine-Learning-Algorithms-Extensions/blob/master/SVM%20Soft%20Margin%20with%20scikit-learn.ipynb), [NumPy](https://github.com/jmg764/Machine-Learning-Algorithms-Extensions/blob/master/SVM%20Soft%20Margin%20with%20NumPy.ipynb))

## Extension 1: Neural Network with Softmax

In class, we discussed setting up a neural network using sigmoid activation functions for both the hidden and output layers. However, using sigmoid as the activation function for the output layer may not be suitable for a classification problem that involves only one right answer such as classification of handwritten digits using the MNIST dataset. In the output layer, for each linear sum produced by combining the output of the previous layer with the corresponding weights and biases (z), sigmoid produces independent probabilities. Since sigmoid is applied to each raw output value separately, the resulting values are not constrained to sum to one which makes sigmoid appropriate for classification problems that involve more than one right answer. On the other hand, softmax can be used in order to obtain a probability distribution for mutually exclusive outputs such as when classifying handwritten digits. Softmax normalizes the probability of each raw output with respect to a summation of all elements thereby making each probability interrelated. In other words, when using softmax, if the probability of one class increases, then the probability of one or more other classes must decrease. Therefore, there can only be one most likely output.

### Keras Implementation

The key component for implementation of softmax is assigning the ```activation``` argument in the ```Dense``` layer to ```'softmax'```:
```python

# Define the keras model with softmax outer layer
model_plus_ext = Sequential()
model_plus_ext.add(Dense(64, activation='sigmoid'))
model_plus_ext.add(Dense(30, activation='sigmoid'))
model_plus_ext.add(Dense(10, activation='softmax'))

# Compile the keras model
model_plus_ext.compile(loss='mean_squared_error', optimizer='sgd', metrics = ['accuracy'])

# Train the keras model
model_plus_ext.fit(X_train, y_v_train, epochs = 150, batch_size = 1)

# Evaluate the keras model
accuracy = model_plus_ext.evaluate(X_test, y_v_test)
``` 

### NumPy Implementation

A softmax function was added to be used on the raw output in the final layer of the neural network. In order to avoid overflow or underflow due to the use of exponentials, np.max(z) was subtracted from z:

```python
def softmax(z):
    return np.exp(z - np.max(z))/np.sum(np.exp(z - np.max(z)), axis=0, keepdims=True) 
```

Softmax was then implemented on the raw output of the final layer as shown below:

```python
def feed_forward_with_softmax(x, W, b, nn_structure):
    a = {1: x} # create a dictionary for holding the a values for all levels
    z = { } # create a dictionary for holding the z values for all the layers
    for l in range(1, len(W) + 1): # for each layer
        node_in = a[l]
        z[l+1] = W[l].dot(node_in) + b[l]  # z^(l+1) = W^(l)*a^(l) + b^(l)
        
        # If the next layer is the output layer, use softmax
        if (z[l+1].shape[0] == nn_structure[-1]):
            a[l+1] = softmax(z[l+1])

        # Else, the next layer is a hidden layer, so use sigmoid 
        else: 
            a[l+1] = f(z[l+1]) # a^(l+1) = f(z^(l+1))

    return a, z
 ```
 
 ### Summary of Accuracies for Extension 1

 |  | **MNIST** | **Fashion-MNIST** |                 
 | :---: | :---: | :---: |   
 | **Keras Baseline** | 77.5 | 74.0 |
 | **Keras Baseline + Extension** | 88.7 | 75.4 |
 | **NumPy Baseline** | 88.5 | 11.3 |            
 | **NumPy Baseline + Extension** | 95.4 | 53.0|  
 
Accuracy was lower overall when using Fashion-MNIST than MNIST. This is likely because Fashion-MNIST consists of 28x28 images of clothes compared to the 8x8 images of handwritten digits in MNIST. The large increase in features makes high accuracy more challenging to achieve for a given neural network. Additionally, each class in the Fashion-MNIST dataset consists of a category of clothes which may have greater intra-class variation than in MNIST. For example, the following is a sample from the t-shirt class in Fashion-MNIST:

<p align="center">
<img src="images/fashion_mnist_example.png"  alt="drawing" width="450"/>
</p>
