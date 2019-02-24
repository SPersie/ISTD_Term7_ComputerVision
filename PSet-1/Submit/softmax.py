import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    D = W.shape[0]
    C = W.shape[1]
    N = X.shape[0]

    for i in range(N):
    	# Compute score and probability
    	score = X[i, :].dot(W)
    	score_exp = np.exp(score)
    	probability = score_exp / np.sum(score_exp)

    	# Update Weight
    	for j in range(D):
    		for k in range(C):
    			if k == y[i]:
    				dW[j, k] += X.T[j, i] * (probability[k] - 1)
    			else:
    				dW[j, k] += X.T[j, i] * probability[k]

    	# Add loss
    	loss += -np.log(probability[y[i]])

    loss /= N
    loss += 0.5 * reg * np.sum(W ** 2)

    dW /= N
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[0]

    score = np.dot(X, W)
    score -= np.max(score, axis=1, keepdims=True)
    score_exp = np.exp(score)
    probability = score_exp / np.sum(score_exp)
    correct = probability[range(N), y]

    loss = np.sum(-np.log(correct))
    loss /= N
    loss += 0.5 * reg * np.sum(W ** 2)

    probability[range(N), y] -= 1
    dW = X.T.dot(probability)
    dW /= N

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW

