import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    # data: N x Dx, W1: Dx x H, b: 1 x H 
    a = data.dot(W1) + b1
    h = sigmoid(a)
    # h: N x H, W2: H x Dy, b2: 1 x Dy
    t = h.dot(W2) + b2
    y_hat = softmax(t)
    # y_hat: N x Dy, labels: N x Dy (as int)
    probs = labels * y_hat
    cost = np.sum(-np.log(probs.sum(axis=1)))
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    # obtain the softmax gradient
    dJdt = (y_hat - labels) # N x Dy

    # b2 grad is sum along each index of the Dy vectors
    gradb2 = np.sum(dJdt, 0) 

    # h: N x H, dJdt: N x Dy
    gradW2 = h.T.dot(dJdt) # H x Dy

    # dJdt: N x Dy, W2: H x Dy
    dJdh = dJdt.dot(W2.T)
    # h: N x H
    dhda = sigmoid_grad(h)

    # data: N x Dx, dhda: N x H, DJdh: N x H
    gradW1 = data.T.dot(dhda * dJdh)
    
    # dhda: N x H, DJdh: N x H
    gradb1 = np.sum(dhda * dJdh, 0)
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    print "Actually, no need - you're awesome"
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()