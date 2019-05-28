# This source is originally from https://github.com/d2l-ai/d2l-en

import sys
sys.path.insert(0, '..')

import d2l
from mxnet import nd

def corr2d_multi_in(X, K):
#    for x, k in zip(X, K):
#        print(d2l.corr2d(x, k))
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])

X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print("X.shape: ", X.shape)

print("corr2d_multi_in: ", corr2d_multi_in(X, K))
print("corr2d_multi_in.shape: ", corr2d_multi_in(X, K).shape)

K = nd.stack(K, K + 1, K + 2)
print("K.shape: ", K.shape)
#print("K: ", K)

def corr2d_multi_in_out(X, K):
#    for k in K:
#        print(corr2d_multi_in(X, k))
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])

print("corr2d_multi_in_out: ", corr2d_multi_in_out(X, K))
print("corr2d_multi_in_out.shape: ", corr2d_multi_in_out(X, K).shape)


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)
    return Y.reshape((c_o, h, w))

X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print("Y1: ", Y1)
print("Y2: ", Y2)

(Y1 - Y2).norm().asscalar() < 1e-6
print((Y1 - Y2).norm().asscalar() < 1e-6)
