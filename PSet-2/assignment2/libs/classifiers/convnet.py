import numpy as np

from libs.layers import *
from libs.fast_layers import *
from libs.layer_utils import *
# from layers import *
# from fast_layers import *
# from layer_utils import *

class ConvNet(object):
    """
    INPUT -> [CONV -> SPATIAL_BN -> RELU -> POOL_2x2]*2 
                                    -> [FC_100 -> BN -> RELU]*1 -> FC_10 -> SOFTMAX
    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
        
        self.params = {}
        self.reg = reg
        self.dtype = dtype
    
        D1, W1, H1 = input_dim
        P = (filter_size - 1) / 2
        stride = 1
        
        # conv - pool output dim
        pooling_size = 2
        D2, W2, H2 = conv_pool_output_dim(input_dim, num_filters, filter_size, stride, pooling_size)
        
        # conv - pool output dim
        D3, W3, H3 = conv_output_dim((D2, W2, H2), num_filters, filter_size, stride)
        
        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, D1, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        
        self.params['gamma1'] = np.ones(D2)
        self.params['beta1'] = np.zeros(D2)
        
        self.params['W2'] = np.random.normal(0, weight_scale, (num_filters, D2, filter_size, filter_size))
        self.params['b2'] = np.zeros(num_filters)
        
        self.params['gamma2'] = np.ones(D3)
        self.params['beta2'] = np.zeros(D3)
        
        self.params['W3'] = np.random.normal(0, weight_scale, (D3 * W3 * H3, hidden_dim))
        self.params['b3'] = np.zeros(hidden_dim)
        
        self.params['gamma3'] = np.ones(hidden_dim)
        self.params['beta3'] = np.ones(hidden_dim)
        
        self.params['W4'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b4'] = np.zeros(num_classes)
    
        self.bn_params = []
        self.bn_params = [{'mode': 'train'} for i in xrange(0, 3)]
            
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
            
    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        
        bn_params = self.bn_params
        for bn_param in bn_params:
            bn_param[mode] = mode
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        params1 = conv_param, pool_param, bn_params[0]
        out1, cache1 = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, params1)
        
        out2, cache2 = conv_bn_relu_forward(out1, W2, b2, gamma2, beta2, conv_param, bn_params[1])
        
        out3, cache3 = affine_bn_relu_forward(out2, W3, b3, gamma3, beta3, bn_params[2])
        
        out4, cache4 = affine_forward(out3, W4, b4)
        scores = out4
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        if y is None:
            return scores
        
        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout4 = softmax_loss(scores, y)
        
        loss += 0.5 * self.reg * np.sum(np.square(W1))
        loss += 0.5 * self.reg * np.sum(np.square(W2))
        loss += 0.5 * self.reg * np.sum(np.square(W3))
        loss += 0.5 * self.reg * np.sum(np.square(W4))
        
        dx4, dw4, db4                   = affine_backward(dout4, cache4)
        dx3, dw3, db3, dgamma3, dbeta3  = affine_bn_relu_backward(dx4, cache3)
        dx2, dw2, db2, dgamma2, dbeta2  = conv_bn_relu_backward(dx3, cache2)
        dx1, dw1, db1, dgamma1, dbeta1  = conv_bn_relu_pool_backward(dx2, cache1)
        
        grads['gamma1'] = dgamma1
        grads['beta1'] = dbeta1
        grads['gamma2'] = dgamma2
        grads['beta2'] = dbeta2
        grads['gamma3'] = dgamma3
        grads['beta3'] = dbeta3
        grads['W4'] = dw4 + self.reg * W4
        grads['b4'] = db4
        grads['W3'] = dw3 + self.reg * W3
        grads['b3'] = db3
        grads['W2'] = dw2 + self.reg * W2
        grads['b2'] = db2
        grads['W1'] = dw1 + self.reg * W1
        grads['b1'] = db1
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        return loss, grads