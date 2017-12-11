import numpy as np

# 4x3
x = np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]])

# 4x1
y = np.array([[0,1,1,0]]).T

# printing inputs and respective labels
print 'x:\n', x
print 'y:\n', y

# initialising epochs
epochs = 10000

# initialising synaptic weights
weights_12 = 2*np.random.random((3,4)) - 1 # 3x4
weights_23 = 2*np.random.random((4,5)) - 1 # 4x5
weights_34 = 2*np.random.random((5,1)) - 1 # 5x1

# defining sigmoid function
def sigmoid(x, derivation=False):
    if(derivation):
        return x*(1-x)
    return 1/(1+np.exp(-x))

for i in xrange(epochs):
    # doing forward propagation
    layer_1 = x #inputs go here (4x3)
    layer_2 = sigmoid(np.dot(layer_1, weights_12)) # 4x4
    layer_3 = sigmoid(np.dot(layer_2, weights_23)) # 4x5
    layer_4 = sigmoid(np.dot(layer_3, weights_34)) # outputs come here (4x1)

    # Back propagation starts..
    # finding error for output layer
    layer_4_error = y - layer_4 # 4x1

    #printing mean error
    if(i%1000 == 0):
        print 'Error: ' + str(np.mean(np.abs(layer_4_error)))

    # error weighted derivatives for output layer
    layer_4_delta = layer_4_error * sigmoid(layer_4, derivation=True) # 4x1

    # finding error and weight adjustments for layer 3 and later updating the weights
    layer_3_error = np.dot(layer_4_delta, weights_34.T) # (4x1) x (1x5) = 4x5
    weights_34_adjustments = np.dot(layer_3.T, layer_4_delta) # (5x4) x (4x1) = 5x1
    weights_34 += weights_34_adjustments # 5x1

    # error weighted derivatives for layer 3
    layer_3_delta = layer_3_error * sigmoid(layer_3, derivation=True) # 4x5

    # finding error and weight adjustments for layer 2 and later updating the weights
    layer_2_error = np.dot(layer_3_delta, weights_23.T) # (4x5) x (5x4) = 4x4
    weights_23_adjustments = np.dot(layer_2.T, layer_3_delta) # (4x4) x (4x5) = 4x5
    weights_23 += weights_23_adjustments # 4x5

    # error weighted derivatives for layer 2
    layer_2_delta = layer_2_error * sigmoid(layer_2, derivation=True) # 4x4

    # finding weight adjustments for layer 1 and later updating the weights
    weights_12_adjustments = np.dot(layer_1.T, layer_2_delta) # (3x4) x (4x4) = 3x4
    weights_12 += weights_12_adjustments # 3x4
