# import numpy as np
# import math
#
# def affine_forward(x, w, b):
#     """
#     Computes the forward pass for an affine (fully-connected) layer.
#
#     The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
#     examples, where each example x[i] has shape (d_1, ..., d_k). We will
#     reshape each input into a vector of dimension D = d_1 * ... * d_k, and
#     then transform it to an output vector of dimension M.
#
#     Inputs:
#     - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
#     - w: A numpy array of weights, of shape (D, M)
#     - b: A numpy array of biases, of shape (M,)
#
#     Returns a tuple of:
#     - out: output, of shape (N, M)
#     - cache: (x, w, b)
#     """
#     out = None
#     #############################################################################
#     # TODO: Implement the affine forward pass. Store the result in out. You     #
#     # will need to reshape the input into rows.                                 #
#     #############################################################################
#     N = x.shape[0]
#     x_p = x.reshape(N, -1)
#     out = x_p.dot(w) + b
#
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#     cache = (x, w, b)
#     return out, cache
#
#
# def affine_backward(dout, cache):
#     """
#     Computes the backward pass for an affine layer.
#
#     Inputs:
#     - dout: Upstream derivative, of shape (N, M)
#     - cache: Tuple of:
#       - x: Input data, of shape (N, d_1, ... d_k)
#       - w: Weights, of shape (D, M)
#
#     Returns a tuple of:
#     - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
#     - dw: Gradient with respect to w, of shape (D, M)
#     - db: Gradient with respect to b, of shape (M,)
#     """
#     x, w, b = cache
#     dx, dw, db = None, None, None
#     #############################################################################
#     # TODO: Implement the affine backward pass.                                 #
#     #############################################################################
#     dx = dout.dot(w.T)
#     dx = dx.reshape(x.shape)
#
#     N = x.shape[0]
#     x_p = x.reshape(N, -1)
#     dw = x_p.T.dot(dout)
#
#     db = np.sum(dout, axis=0)
#
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#     return dx, dw, db
#
#
# def relu_forward(x):
#     """
#     Computes the forward pass for a layer of rectified linear units (ReLUs).
#
#     Input:
#     - x: Inputs, of any shape
#
#     Returns a tuple of:
#     - out: Output, of the same shape as x
#     - cache: x
#     """
#     out = None
#     #############################################################################
#     # TODO: Implement the ReLU forward pass.                                    #
#     #############################################################################
#     out = np.copy(x)
#     out[out < 0] = 0
#
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#     cache = x
#     return out, cache
#
#
# def relu_backward(dout, cache):
#     """
#     Computes the backward pass for a layer of rectified linear units (ReLUs).
#
#     Input:
#     - dout: Upstream derivatives, of any shape
#     - cache: Input x, of same shape as dout
#
#     Returns:
#     - dx: Gradient with respect to x
#     """
#     dx, x = None, cache
#     #############################################################################
#     # TODO: Implement the ReLU backward pass.                                   #
#     #############################################################################
#     dx = np.copy(dout)
#     dx[x < 0] = 0
#
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#     return dx
#
# def batchnorm_forward(x, gamma, beta, bn_param):
#   """
#   Forward pass for batch normalization.
#
#   During training the sample mean and (uncorrected) sample variance are
#   computed from minibatch statistics and used to normalize the incoming data.
#   During training we also keep an exponentially decaying running mean of the mean
#   and variance of each feature, and these averages are used to normalize data
#   at test-time.
#   At each timestep we update the running averages for mean and variance using
#   an exponential decay based on the momentum parameter:
#   running_mean = momentum * running_mean + (1 - momentum) * sample_mean
#   running_var = momentum * running_var + (1 - momentum) * sample_var
#   Note that the batch normalization paper suggests a different test-time
#   behavior: they compute sample mean and variance for each feature using a
#   large number of training images rather than using a running average. For
#   this implementation we have chosen to use running averages instead since
#   they do not require an additional estimation step; the torch7 implementation
#   of batch normalization also uses running averages.
#   Input:
#   - x: Data of shape (N, D)
#   - gamma: Scale parameter of shape (D,)
#   - beta: Shift paremeter of shape (D,)
#   - bn_param: Dictionary with the following keys:
#     - mode: 'train' or 'test'; required
#     - eps: Constant for numeric stability
#     - momentum: Constant for running mean / variance.
#     - running_mean: Array of shape (D,) giving running mean of features
#     - running_var Array of shape (D,) giving running variance of features
#   Returns a tuple of:
#   - out: of shape (N, D)
#   - cache: A tuple of values needed in the backward pass
#   """
#   mode = bn_param['mode']
#   # print('This is to prove that i run batchnorm')
#   eps = bn_param.get('eps', 1e-5)
#   momentum = bn_param.get('momentum', 0.9)
#
#   N, D = x.shape
#   running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
#   running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
#
#   out, cache = None, None
#   if mode == 'train':
#     #############################################################################
#     # TODO: Implement the training-time forward pass for batch normalization.   #
#     # Use minibatch statistics to compute the mean and variance, use these      #
#     # statistics to normalize the incoming data, and scale and shift the        #
#     # normalized data using gamma and beta.                                     #
#     #                                                                           #
#     # You should store the output in the variable out. Any intermediates that   #
#     # you need for the backward pass should be stored in the cache variable.    #
#     #                                                                           #
#     # You should also use your computed sample mean and variance together with  #
#     # the momentum variable to update the running mean and running variance,    #
#     # storing your result in the running_mean and running_var variables.        #
#     #############################################################################
#     batch_mean = np.mean(x, axis = 0)
#     batch_var = np.var(x, axis = 0)
#
#     x_hat = (x-batch_mean) / np.sqrt(batch_var + eps)
#     out = gamma * x_hat + beta
#
#
#     running_mean = momentum * running_mean + (1 - momentum) * batch_mean
#     running_var = momentum * running_var + (1 - momentum) * batch_var
#
#     cache = (gamma, beta, batch_mean, batch_var, x_hat, x, eps)
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#   elif mode == 'test':
#     #############################################################################
#     # TODO: Implement the test-time forward pass for batch normalization. Use   #
#     # the running mean and variance to normalize the incoming data, then scale  #
#     # and shift the normalized data using gamma and beta. Store the result in   #
#     # the out variable.                                                         #
#     #############################################################################
#     x_normalized = (x - running_mean)/np.sqrt(running_var +eps)
#     out = gamma*x_normalized + beta
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#   else:
#     raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
#
#   # Store the updated running means back into bn_param
#   bn_param['running_mean'] = running_mean
#   bn_param['running_var'] = running_var
#
#   return out, cache
#
#
#
# def batchnorm_backward(dout, cache):
#     """
#     Backward pass for batch normalization.
#
#     For this implementation, you should write out a computation graph for
#     batch normalization on paper and propagate gradients backward through
#     intermediate nodes.
#
#     Inputs:
#     - dout: Upstream derivatives, of shape (N, D)
#     - cache: Variable of intermediates from batchnorm_forward.
#
#     Returns a tuple of:
#     - dx: Gradient with respect to inputs x, of shape (N, D)
#     - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
#     - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
#     """
#     dx, dgamma, dbeta = None, None, None
#     gamma, beta, batch_mean, batch_var, x_hat, x, eps = cache
#     #############################################################################
#     # TODO: Implement the backward pass for batch normalization. Store the      #
#     # results in the dx, dgamma, and dbeta variables.                           #
#     #############################################################################
#     N = x.shape[0]
#     dbeta = np.sum(dout, axis=0)
#     dgamma = np.sum(x_hat * dout, axis=0)
#     dx_hat = gamma * dout
#     dbatch_var = np.sum(-1.0/2*dx_hat*(x-batch_mean)/(batch_var+eps)**(3.0/2), axis =0)
#     dbatch_mean = np.sum(-1/np.sqrt(batch_var+eps)* dx_hat, axis = 0) + 1.0/N*dbatch_var *np.sum(-2*(x-batch_mean), axis = 0)
#     dx = 1/np.sqrt(batch_var+eps)*dx_hat + dbatch_var*2.0/N*(x-batch_mean) + 1.0/N*dbatch_mean
#     pass
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#
#     return dx, dgamma, dbeta
#
#
# def batchnorm_backward_alt(dout, cache):
#     """
#     Alternative backward pass for batch normalization.
#
#     For this implementation you should work out the derivatives for the batch
#     normalizaton backward pass on paper and simplify as much as possible. You
#     should be able to derive a simple expression for the backward pass.
#
#     Note: This implementation should expect to receive the same cache variable
#     as batchnorm_backward, but might not use all of the values in the cache.
#
#     Inputs / outputs: Same as batchnorm_backward
#     """
#     dx, dgamma, dbeta = None, None, None
#     #############################################################################
#     # TODO: Implement the backward pass for batch normalization. Store the      #
#     # results in the dx, dgamma, and dbeta variables.                           #
#     #                                                                           #
#     # After computing the gradient with respect to the centered inputs, you     #
#     # should be able to compute gradients with respect to the inputs in a       #
#     # single statement; our implementation fits on a single 80-character line.  #
#     #############################################################################
#     gamma, beta, batch_mean, batch_var, x_hat, x, eps = cache
#     N = x.shape[0]
#     dbeta = np.sum(dout, axis=0)
#     dgamma = np.sum(x_hat*dout, axis = 0)
#     dx_hat = gamma* dout
#     dbatch_var = np.sum(-1.0/2*dx_hat*x_hat/(batch_var+eps), axis =0)
#     ### drop the second term which simplfies to zero
#     dbatch_mean = np.sum(-1/np.sqrt(batch_var+eps)* dx_hat, axis = 0)
#     dx = 1/np.sqrt(batch_var+eps)*dx_hat + dbatch_var*2.0/N*(x-batch_mean) + 1.0/N*dbatch_mean
#     pass
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#
#     return dx, dgamma, dbeta
#
#
# def dropout_forward(x, dropout_param):
#     """
#     Performs the forward pass for (inverted) dropout.
#
#     Inputs:
#     - x: Input data, of any shape
#     - dropout_param: A dictionary with the following keys:
#       - p: Dropout parameter. We drop each neuron output with probability p.
#       - mode: 'test' or 'train'. If the mode is train, then perform dropout;
#         if the mode is test, then just return the input.
#       - seed: Seed for the random number generator. Passing seed makes this
#         function deterministic, which is needed for gradient checking but not in
#         real networks.
#
#     Outputs:
#     - out: Array of the same shape as x.
#     - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
#       mask that was used to multiply the input; in test mode, mask is None.
#     """
#     p, mode = dropout_param['p'], dropout_param['mode']
#     if 'seed' in dropout_param:
#         np.random.seed(dropout_param['seed'])
#
#     mask = None
#     out = None
#
#     if mode == 'train':
#         ###########################################################################
#         # TODO: Implement the training phase forward pass for inverted dropout.   #
#         # Store the dropout mask in the mask variable.                            #
#         ###########################################################################
#         a, b = x.shape
#         temp = np.random.rand(a, b) < (1 - p)
#         mask = temp / (1 - p)
#         out = x * mask
#
#         pass
#         ###########################################################################
#         #                            END OF YOUR CODE                             #
#         ###########################################################################
#     elif mode == 'test':
#         ###########################################################################
#         # TODO: Implement the test phase forward pass for inverted dropout.       #
#         ###########################################################################
#         out = x
#
#         pass
#         ###########################################################################
#         #                            END OF YOUR CODE                             #
#         ###########################################################################
#
#     cache = (dropout_param, mask)
#     out = out.astype(x.dtype, copy=False)
#
#     return out, cache
#
# def dropout_backward(dout, cache):
#     """
#     Perform the backward pass for (inverted) dropout.
#
#     Inputs:
#     - dout: Upstream derivatives, of any shape
#     - cache: (dropout_param, mask) from dropout_forward.
#     """
#     dropout_param, mask = cache
#     mode = dropout_param['mode']
#
#     dx = None
#     if mode == 'train':
#         ###########################################################################
#         # TODO: Implement the training phase backward pass for inverted dropout.  #
#         ###########################################################################
#         dx = mask * dout
#
#         ###########################################################################
#         #                            END OF YOUR CODE                             #
#         ###########################################################################
#     elif mode == 'test':
#         dx = dout
#     return dx
#
#
# def conv_forward_naive(x, w, b, conv_param):
#     """
#     A naive implementation of the forward pass for a convolutional layer.
#
#     The input consists of N data points, each with C channels, height H and width
#     W. We convolve each input with F different filters, where each filter spans
#     all C channels and has height HH and width HH.
#
#     Input:
#     - x: Input data of shape (N, C, H, W)
#     - w: Filter weights of shape (F, C, HH, WW)
#     - b: Biases, of shape (F,)
#     - conv_param: A dictionary with the following keys:
#       - 'stride': The number of pixels between adjacent receptive fields in the
#         horizontal and vertical directions.
#       - 'pad': The number of pixels that will be used to zero-pad the input.
#
#     Returns a tuple of:
#     - out: Output data, of shape (N, F, H', W') where H' and W' are given by
#       H' = 1 + (H + 2 * pad - HH) / stride
#       W' = 1 + (W + 2 * pad - WW) / stride
#     - cache: (x, w, b, conv_param)
#     """
#
#     cache = None
#     #############################################################################
#     # TODO: Implement the convolutional forward pass.                           #
#     # Hint: you can use the function np.pad for padding.                        #
#     #############################################################################
#     N, C, H, W = x.shape
#     F, C_, HH, WW = w.shape
#     stride = conv_param['stride']
#     pad = conv_param['pad']
#     H_opt = ((H + 2 * pad - HH) / stride) + 1
#     # print(H_opt)
#     H_opt = math.ceil(H_opt)
#     W_opt = ((W + 2 * pad - WW) / stride) + 1
#     # print(W_opt)
#     W_opt = math.ceil(W_opt)
#     opt = np.zeros((N, F, H_opt, W_opt))
#
#     ### Input padding
#     temp_x = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
#     for i in range(N):
#         for j in range(C):
#             temp_x[i, j] = np.pad(x[i, j], [1, 1],
#                                 mode='constant', constant_values=(0,0))
#
#     ### Forward convolution
#     for i in range(N):
#         for j in range(H_opt):
#             for k in range(W_opt):
#                 for f in range(F):
#                     ### x and filter to fo convolution
#                     pend_x = temp_x[i, :, j*stride:j*stride+HH, k*stride:k*stride+WW]
#                     pend_filter = w[f]
#                     ### compute for each filter
#                     opt[i, f, j, k] = np.sum(pend_x * pend_filter)
#                 ### add bias
#                 opt[i, :, j, k] = opt[i, :, j, k] + b
#     out = opt
#
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#     cache = (x, w, b, conv_param)
#     return out, cache
#
#
# def conv_backward_naive(dout, cache):
#     """
#     A naive implementation of the backward pass for a convolutional layer.
#
#     Inputs:
#     - dout: Upstream derivatives.
#     - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
#
#     Returns a tuple of:
#     - dx: Gradient with respect to x
#     - dw: Gradient with respect to w
#     - db: Gradient with respect to b
#     """
#     dx, dw, db = None, None, None
#     x, w, b, conv_param = cache
#
#     N, C, H, W = x.shape
#     F, C, HH, WW = w.shape
#
#     stride = conv_param['stride']
#     pad = conv_param['pad']
#
#     # The shape of output layer.
#     H_prime = int(1 + (H + 2 * pad - HH) / stride)
#     W_prime = int(1 + (W + 2 * pad - WW) / stride)
#     x_pad = np.lib.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),\
#                         'constant', constant_values=(0))
#     #############################################################################
#     # TODO: Implement the convolutional backward pass.                          #
#     #############################################################################
#     db = np.zeros((F))
#     for n in range(N):
#         for hp in range(H_prime):
#             for wp in range(W_prime):
#                 db = db + dout[n, :, hp, wp]
#
#     dw = np.zeros((F, C, HH, WW))
#     dx_pad = np.zeros(x_pad.shape)
#
#     for n in range(N):
#         for f in range(F):
#             for h in range(H_prime):
#                 for wp in range(W_prime):
#                     temp_x = x_pad[n, :, h*stride:h*stride+HH, wp*stride:wp*stride+WW]
#                     dw[f] = dw[f] + dout[n, f, h, wp] * temp_x
#                     dx_pad[n, :, h*stride:h*stride+HH, wp*stride:wp*stride+WW] += w[f] * dout[n, f, h, wp]
#
#     dx = dx_pad[:, :, 1:H+1, 1:W+1]
#     pass
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#     return dx, dw, db
#
#
# def max_pool_forward_naive(x, pool_param):
#     """
#     A naive implementation of the forward pass for a max pooling layer.
#
#     Inputs:
#     - x: Input data, of shape (N, C, H, W)
#     - pool_param: dictionary with the following keys:
#       - 'pool_height': The height of each pooling region
#       - 'pool_width': The width of each pooling region
#       - 'stride': The distance between adjacent pooling regions
#
#     Returns a tuple of:
#     - out: Output data
#     - cache: (x, pool_param)
#     """
#     out = None
#
#     #############################################################################
#     # TODO: Implement the max pooling forward pass                              #
#     #############################################################################
#     N, C, H, W = x.shape
#     pool_height = pool_param['pool_height']
#     pool_width = pool_param['pool_width']
#     stride = pool_param['stride']
#     pad = 0
#     H_prime = int(1 + (H + 2 * pad - pool_height) / stride)
#     W_prime = int(1 + (W + 2 * pad - pool_width) / stride)
#     out = np.zeros((N, C, H_prime, W_prime))
#
#     for n in range(N):
#         for c in range(C):
#             for hp in range(H_prime):
#                 for wp in range(W_prime):
#                     # print(x.shape)
#                     out[n, c, hp, wp] = np.max(x[n, c, hp*stride:hp*stride+pool_height, wp*stride:wp*stride+pool_width])
#
#     # pass
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#     cache = (x, pool_param)
#     return out, cache
#
#
# def max_pool_backward_naive(dout, cache):
#     """
#     A naive implementation of the backward pass for a max pooling layer.
#
#     Inputs:
#     - dout: Upstream derivatives
#     - cache: A tuple of (x, pool_param) as in the forward pass.
#
#     Returns:
#     - dx: Gradient with respect to x
#     """
#     dx = None
#
#     #############################################################################
#     # TODO: Implement the max pooling backward pass                             #
#     #############################################################################
#     N, C, H_prime, W_prime = dout.shape
#     x, pool_param = cache
#     dx = np.zeros(x.shape)
#     pool_height = pool_param['pool_height']
#     pool_width = pool_param['pool_width']
#     stride = pool_param['stride']
#
#     for n in range(N):
#         for c in range(C):
#             for hp in range(H_prime):
#                 for wp in range(W_prime):
#                     temp_x = x[n, c, hp*stride:hp*stride+pool_height, wp*stride:wp*stride+pool_width]
#                     temp_max = np.max(temp_x)
#                     for ph in range(pool_height):
#                         for pw in range(pool_width):
#                             if temp_x[ph, pw] == temp_max:
#                                 dx[n, c, hp*stride+ph, wp*stride+pw] += dout[n, c, hp, wp]
#
#     # pass
#
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#     return dx
#
#
# def spatial_batchnorm_forward(x, gamma, beta, bn_param):
#     """
#     Computes the forward pass for spatial batch normalization.
#
#     Inputs:
#     - x: Input data of shape (N, C, H, W)
#     - gamma: Scale parameter, of shape (C,)
#     - beta: Shift parameter, of shape (C,)
#     - bn_param: Dictionary with the following keys:
#       - mode: 'train' or 'test'; required
#       - eps: Constant for numeric stability
#       - momentum: Constant for running mean / variance. momentum=0 means that
#         old information is discarded completely at every time step, while
#         momentum=1 means that new information is never incorporated. The
#         default of momentum=0.9 should work well in most situations.
#       - running_mean: Array of shape (D,) giving running mean of features
#       - running_var Array of shape (D,) giving running variance of features
#
#     Returns a tuple of:
#     - out: Output data, of shape (N, C, H, W)
#     - cache: Values needed for the backward pass
#     """
#     out, cache = None, None
#
#     #############################################################################
#     # TODO: Implement the forward pass for spatial batch normalization.         #
#     #                                                                           #
#     # HINT: You can implement spatial batch normalization using the vanilla     #
#     # version of batch normalization defined above. Your implementation should  #
#     # be very short; ours is less than five lines.                              #
#     #############################################################################
#     pass
#
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#
#
#
#     return out, cache
#
#
# def spatial_batchnorm_backward(dout, cache):
#     """
#     Computes the backward pass for spatial batch normalization.
#
#     Inputs:
#     - dout: Upstream derivatives, of shape (N, C, H, W)
#     - cache: Values from the forward pass
#
#     Returns a tuple of:
#     - dx: Gradient with respect to inputs, of shape (N, C, H, W)
#     - dgamma: Gradient with respect to scale parameter, of shape (C,)
#     - dbeta: Gradient with respect to shift parameter, of shape (C,)
#     """
#
#     #############################################################################
#     # TODO: Implement the backward pass for spatial batch normalization.        #
#     #                                                                           #
#     # HINT: You can implement spatial batch normalization using the vanilla     #
#     # version of batch normalization defined above. Your implementation should  #
#     # be very short; ours is less than five lines.                              #
#     #############################################################################
#     pass
#     #############################################################################
#     #                             END OF YOUR CODE                              #
#     #############################################################################
#     return dx, dgamma, dbeta
#
#
# # def svm_loss(x, y):
# #     """
# #     Computes the loss and gradient using for multiclass SVM classification.
#
# #     Inputs:
# #     - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
# #       for the ith input.
# #     - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
# #       0 <= y[i] < C
#
# #     Returns a tuple of:
# #     - loss: Scalar giving the loss
# #     - dx: Gradient of the loss with respect to x
# #     """
# #     N = x.shape[0]
# #     correct_class_scores = x[np.arange(N), y]
# #     margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
# #     margins[np.arange(N), y] = 0
# #     loss = np.sum(margins) / N
# #     num_pos = np.sum(margins > 0, axis=1)
# #     dx = np.zeros_like(x)
# #     dx[margins > 0] = 1
# #     dx[np.arange(N), y] -= num_pos
# #     dx /= N
# #     return loss, dx
#
#
# def softmax_loss(x, y):
#     """
#     Computes the loss and gradient for softmax classification.
#
#     Inputs:
#     - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
#       for the ith input.
#     - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
#       0 <= y[i] < C
#
#     Returns a tuple of:
#     - loss: Scalar giving the loss
#     - dx: Gradient of the loss with respect to x
#     """
#     probs = np.exp(x - np.max(x, axis=1, keepdims=True))
#     probs /= np.sum(probs, axis=1, keepdims=True)
#     N = x.shape[0]
#     loss = -np.sum(np.log(probs[np.arange(N), y])) / N
#     dx = probs.copy()
#     dx[np.arange(N), y] -= 1
#     dx /= N
#     return loss, dx




import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.
  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.
  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = x.shape[0]
  x_temp = x.reshape(N,-1)
  out = x_temp.dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.
  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  db = np.sum(dout, axis = 0)
  x_temp = x.reshape(x.shape[0],-1)
  dw = x_temp.T.dot(dout)
  dx = dout.dot(w.T).reshape(x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).
  Input:
  - x: Inputs, of any shape
  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.copy(x)
  out[out<0] = 0
  #############################################################################
  #             END OF YOUR CODE                                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).
  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout
  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = np.copy(dout)
  dx[x<0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.
  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:
  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var
  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.
  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_mean = np.mean(x, axis = 0)
    sample_var = np.var(x, axis = 0)

    x_normalized = (x-sample_mean) / np.sqrt(sample_var + eps)
    out = gamma*x_normalized + beta


    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    cache = (x, sample_mean, sample_var, x_normalized, beta, gamma, eps)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x_normalized = (x - running_mean)/np.sqrt(running_var +eps)
    out = gamma*x_normalized + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################

  (x, sample_mean, sample_var, x_normalized, beta, gamma, eps) = cache
  N = x.shape[0]
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(x_normalized*dout, axis = 0)
  dx_normalized = gamma* dout
  dsample_var = np.sum(-1.0/2*dx_normalized*(x-sample_mean)/(sample_var+eps)**(3.0/2), axis =0)
  dsample_mean = np.sum(-1/np.sqrt(sample_var+eps)* dx_normalized, axis = 0) + 1.0/N*dsample_var *np.sum(-2*(x-sample_mean), axis = 0)
  dx = 1/np.sqrt(sample_var+eps)*dx_normalized + dsample_var*2.0/N*(x-sample_mean) + 1.0/N*dsample_mean


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.

  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.

  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.

  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  (x, sample_mean, sample_var, x_normalized, beta, gamma, eps) = cache
  N = x.shape[0]
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(x_normalized*dout, axis = 0)
  dx_normalized = gamma* dout
  dsample_var = np.sum(-1.0/2*dx_normalized*x_normalized/(sample_var+eps), axis =0)
  dsample_mean = np.sum(-1/np.sqrt(sample_var+eps)* dx_normalized, axis = 0) # drop the second term which simplfies to zero
  dx = 1/np.sqrt(sample_var+eps)*dx_normalized + dsample_var*2.0/N*(x-sample_mean) + 1.0/N*dsample_mean


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.
  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.
  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################

    # the lecture slide and here uses different definitions of p
    # Slide: p is the probablity of a neuron to be kept
    # Here: p is the probability of a neuron to be drop

    [N,D] = x.shape
    mask = (np.random.rand(N,D) < (1-p))/(1-p)
    out = x*mask

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.
  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']

  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = mask*dout
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.
  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.
  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.
  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros((N,F,H_out,W_out))

  # Pad the input
  x_pad = np.zeros((N,C,H+2*pad,W+2*pad))
  for n in range(N):
    for c in range(C):
      x_pad[n,c] = np.pad(x[n,c],(1,1),'constant', constant_values=(0,0))

  for n in range(N):
      for i in range(H_out):
        for j in range(W_out):
          for f in range(F):
            current_x_matrix = x_pad[n,:, i*stride: i*stride+HH, j*stride:j*stride+WW]
            current_filter = w[f]
            out[n,f,i,j] = np.sum(current_x_matrix*current_filter)
          out[n,:,i,j] = out[n,:,i,j]+b

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.
  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  _,_,H_out,W_out = dout.shape

  x_pad = np.zeros((N,C,H+2,W+2))
  for n in range(N):
    for c in range(C):
      x_pad[n,c] = np.pad(x[n,c],(1,1),'constant', constant_values=(0,0))

  db = np.zeros((F))
  for n in range(N):
    for i in range(H_out):
      for j in range(W_out):
          db = db + dout[n,:,i,j]

  dw = np.zeros(w.shape)
  dx_pad = np.zeros(x_pad.shape)

  for n in range(N):
    for f in range(F):
      for i in range(H_out):
        for j in range(W_out):
          current_x_matrix = x_pad[n,:, i*stride: i*stride+HH, j*stride:j*stride+WW]
          dw[f] = dw[f] + dout[n,f,i,j]* current_x_matrix
          dx_pad[n,:, i*stride: i*stride+HH, j*stride:j*stride+WW] += w[f]*dout[n,f,i,j]

  dx = dx_pad[:,:,1:H+1,1:W+1]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.
  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions
  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  H_out = 1 + (H - pool_height) / stride
  W_out = 1 + (W - pool_width) / stride
  out = np.zeros((N,C,H_out,W_out))

  for n in range(N):
    for c in range(C):
      for h in range(H_out):
        for w in range(W_out):
          out[n,c,h,w] = np.max(x[n,c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.
  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.
  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  N,C,H_out,W_out = dout.shape

  dx = np.zeros(x.shape)

# The instruction says ''You don't need to worry about computational efficiency."
# So I did the following...
  for n in range(N):
    for c in range(C):
      for h in range(H_out):
        for w in range(W_out):
          current_matrix= x[n,c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width]
          current_max = np.max(current_matrix)
          for (i,j) in [(i,j) for i in range(pool_height) for j in range(pool_width)]:
            if current_matrix[i,j] == current_max:
               dx[n,c,h*stride+i,w*stride+j] += dout[n,c,h,w]


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  N, C, H, W = x.shape
  temp_output, cache = batchnorm_forward(x.transpose(0,3,2,1).reshape((N*H*W,C)), gamma, beta, bn_param)
  out = temp_output.reshape(N,W,H,C).transpose(0,3,2,1)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  N,C,H,W = dout.shape
  dx_temp, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0,3,2,1).reshape((N*H*W,C)),cache)
  dx = dx_temp.reshape(N,W,H,C).transpose(0,3,2,1)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
