from mxnet import init, nd
from mxnet.gluon import nn

def getnet():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = getnet()

net.initialize()

x = nd.random.uniform(shape=(2, 20))
net(x)  # Forward computation

print(net.collect_params)
print(net.collect_params())

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # The actual initialization logic is omitted here

net = getnet()
print(net.collect_params)
print(net.collect_params())

net.initialize(init=MyInit())

x = nd.random.uniform(shape=(2, 20))
y = net(x)

print(y)


#net.initialize(init=MyInit(), force_reinit=True)

#net = nn.Sequential()
#net.add(nn.Dense(256, in_units=20, activation='relu'))
#net.add(nn.Dense(10, in_units=256))

#net.initialize(init=MyInit())
