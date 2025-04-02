Here's an explanation of each layer in the network along with their functions, default values, and any custom values that were provided:

1. **Convolutional Layer (`layers.Conv1D`):**

   - Function: The Convolutional Neural Network (CNN) uses this layer to apply a set of filters (or kernels) across the input data.
   - Default Values:
    ```
    kernel_size=3,
    activation=None,
    padding='valid',
    dilation_rate=(1, 1),
    depthwise_initializer='ones',
    bias_initializer='zeros'
    ```

    Parameters passed in our model:

    ```python
    kernel_size=5, 
    activation='relu', 
    padding='same'
    ```
   
    **Effects on Performance:** 

    - `kernel_size` is the number of rows and columns to use for each filter. Increasing this can improve performance by allowing more features (edges) to be extracted from input data.
    - `activation='relu'`: This allows ReLU activation, which helps in reducing vanishing gradients and speeds up training process.

2. **Pooling Layer (`layers.MaxPooling1D`):**

    Function: Pooling layers reduce the spatial dimensions of a feature map by taking only every nth element across each row.
   
    Default Values:
    ```
    pool_size=2,
    strides=2,
    padding='valid',
    data_format='channels_last'
    ```

    Parameters passed in our model:

    ```python
    pool_size=2, 
    strides=2   
    ```
   
    **Effects on Performance:**

    - `pool_size` determines how many spatial dimensions to reduce. Using a higher value will result in fewer features extracted but the network will learn more general features.
    - `strides` controls how often each element of the feature map is considered for pooling. This has an effect on the overall size of the output.

3. **Dense Layer (`layers.Dense`):**

    Function: Fully connected layers are used to connect different sets of neurons in a network.

    Default Values:
    ```
    units=None,
    activation='relu',
    use_bias=True
    ```

    Parameters passed in our model:

    ```python
    units=64, 
    activation='relu'
    ```
   
   **Effects on Performance:** 

    - `units` is the number of neurons. Increasing this allows for more parameters and generally helps improve performance by allowing the network to learn a better solution.
    - `activation='relu'`: This enables ReLU activation which can help with training efficiency.

4. **Dropout Layer (`layers.Dropout`):**

    Function: Dropout randomly sets a fraction rate of neurons to zero during training, preventing overfitting and improving robustness to noise in the data.

    Default Values:
    ```
    dropout_rate=0.5,
    training=True
    ```

    Parameters passed in our model:

    ```python
    dropout_rate = 0.3 
    training=False
    ```
   
    **Effects on Performance:** 

    - `dropout_rate` is the fraction of neurons to set to zero during training.
    This rate can be lower, but it increases likelihood that network will generalize well and make accurate predictions.

5. **Global Average Pooling (`layers.GlobalAveragePooling1D`):**

    Function: Global average pooling reduces each feature map across its spatial dimensions by calculating the mean of all elements in each row/column separately.

    Default Values:
    ```
    pool_size=None,
    strides=1
    ```

    Parameters passed in our model:

    ```python 
    Global Average Pooling  
    ```

    **Effects on Performance:** 

    - Global average pooling reduces dimensionality and can be useful for further layers that require one-dimensional input.

6. **Softmax Activation (`layers.Dense(num_classes, activation='softmax')`):**

    Function: The softmax function outputs probabilities of each class in the output space.
   
    Default Values:
    ```
    activation=None
    ```

    Parameters passed in our model:

    ```python 
    activation='softmax'
    ```
   
    **Effects on Performance:** 

    - This helps to prepare for classification by ensuring that all classes have a probability between zero and one.

7. **Optimizer (`'adam'`) & Loss Function (`sparse_categorical_crossentropy`):**

    These are not custom values but rather the default in TensorFlow when using `Keras`. The choice of optimizer, loss function, or both depends on your specific task (e.g., classification, regression) and dataset characteristics.
   
    - `'adam'`: Adam is a popular stochastic gradient descent algorithm used for training neural networks. It tends to perform well for deep learning tasks.

    - `sparse_categorical_crossentropy` : This is suitable when the labels are integers with no fractional components.
    If you were dealing with continuous values, another cross-entropy function like categorical_crossentropy could be more suitable.