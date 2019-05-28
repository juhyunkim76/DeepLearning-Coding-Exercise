# This source is originally from https://github.com/d2l-ai/d2l-en

from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd, gluon, init
display.set_matplotlib_formats('svg')

embedding = 4  # Embedding dimension for autoregressive model
T = 1000  # Generate a total of 1000 points
time = nd.arange(0,T)
x = nd.sin(0.01 * time) + 0.2 * nd.random.normal(shape=(T))

print("x: ", x)
#print(x.shape)
#print(len(x))

plt.plot(time.asnumpy(), x.asnumpy());

features = nd.zeros((T-embedding, embedding))
print("features: ", features)

for i in range(embedding):
#    print("i: ", i)
#    print("T-embedding+i:", T-embedding+i)
    features[:,i] = x[i:T-embedding+i]
labels = x[embedding:]
print("features: ", features)
print("labels: ", labels)

ntrain = 600
train_data = gluon.data.ArrayDataset(features[:ntrain,:], labels[:ntrain])
test_data  = gluon.data.ArrayDataset(features[ntrain:,:], labels[ntrain:])

# Vanilla MLP architecture
def get_net():
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(10, activation='relu'))
    net.add(gluon.nn.Dense(10, activation='relu'))
    net.add(gluon.nn.Dense(1))
    net.initialize(init.Xavier())
    return net

# Least mean squares loss
loss = gluon.loss.L2Loss()

# Simple optimizer using adam, random shuffle and minibatch size 16
def train_net(net, data, loss, epochs, learningrate):
    batch_size = 16
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learningrate})
    data_iter = gluon.data.DataLoader(data, batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(data[:][0]), nd.array(data[:][1]))
        print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
    return net

net = get_net()
net = train_net(net, train_data, loss, 10, 0.01)

l = loss(net(test_data[:][0]), nd.array(test_data[:][1]))
print('test loss: %f' % l.mean().asnumpy())

estimates = net(features)
plt.plot(time.asnumpy(), x.asnumpy(), label='data');
plt.plot(time[embedding:].asnumpy(), estimates.asnumpy(), label='estimate');
plt.legend();

predictions = nd.zeros_like(estimates)
print("predictions: ", predictions)
predictions[:(ntrain-embedding)] = estimates[:(ntrain-embedding)]
for i in range(ntrain-embedding, T-embedding):
#    print("i: ", i)
#    print("i-embedding: ", i-embedding)
#    print("predictions[(i-embedding):i].reshape(1,-1): ", predictions[(i-embedding):i].reshape(1,-1))
    predictions[i] = net(
        predictions[(i-embedding):i].reshape(1,-1)).reshape(1)

plt.plot(time.asnumpy(), x.asnumpy(), label='data');
plt.plot(time[embedding:].asnumpy(), estimates.asnumpy(), label='estimate');
plt.plot(time[embedding:].asnumpy(), predictions.asnumpy(),
         label='multistep');
plt.legend();


k = 33  # Look up to k - embedding steps ahead

features = nd.zeros((T-k, k))
print("features : ", features)
for i in range(embedding):
    features[:,i] = x[i:T-k+i]
print("features : ", features)

for i in range(embedding, k):
    features[:,i] = net(features[:,(i-embedding):i]).reshape((-1))
print("features : ", features)

for i in (4, 8, 16, 32):
    plt.plot(time[i:T-k+i].asnumpy(), features[:,i].asnumpy(),
             label=('step ' + str(i)))
plt.legend();
