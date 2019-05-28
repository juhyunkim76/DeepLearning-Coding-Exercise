# This source is originally from https://github.com/d2l-ai/d2l-en

import sys
sys.path.insert(0, '..')

import d2l
from mxnet.gluon import data as gdata
import sys
import time


mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

feature, label = mnist_train[0]

X, y = mnist_train[0:9]
d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))

batch_size = 256
transformer = gdata.vision.transforms.ToTensor()

if sys.platform.startswith('win'):
    # 0 means no additional processes are needed to speed up the reading of
    # data
    num_workers = 0
else:
    num_workers = 4


train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                batch_size, shuffle=True, num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                batch_size, shuffle=False, num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
