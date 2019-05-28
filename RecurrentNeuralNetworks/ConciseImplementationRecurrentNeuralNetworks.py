# This source is originally from https://github.com/d2l-ai/d2l-en

import sys
sys.path.insert(0, '..')

import d2l
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time

corpus_indices, vocab = d2l.load_data_time_machine()

print("corpus_indices: ", corpus_indices)
print("vocab: ", vocab)
print("len(vocab): ", len(vocab))

num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()

batch_size = 2
state = rnn_layer.begin_state(batch_size=batch_size)
print("state[0].shape: ", state[0].shape)

num_steps = 35
X = nd.random.uniform(shape=(num_steps, batch_size, len(vocab)))
print("X.shape: ", X.shape)
Y, state_new = rnn_layer(X, state)
print("Y.shape: ", Y.shape, "len(state_new): ", len(state_new), "state_new[0].shape: ", state_new[0].shape)
#print("Y: ", Y)
#print("state_new: ", state_new)

class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        # Get the one-hot vector representation by transposing the input to
        # (num_steps, batch_size)
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)

        # The fully connected layer will first change the shape of Y to
        # (num_steps * batch_size, num_hiddens)
        # Its output shape is (num_steps * batch_size, vocab_size)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


def predict_rnn_gluon(prefix, num_chars, model, vocab, ctx):
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [vocab[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        # Forward computation does not require incoming model parameters
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(vocab[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([vocab.idx_to_token[i] for i in output])


ctx = d2l.try_gpu()
model = RNNModel(rnn_layer, len(vocab))
model.initialize(force_reinit=True, ctx=ctx)
print("predict_rnn_gluo: ", predict_rnn_gluon('traveller', 10, model, vocab, ctx))


def grad_clipping_gluon(model, theta, ctx):
    params = [p.data() for p in model.collect_params().values()]
    d2l.grad_clipping(params, theta, ctx)

def train_and_predict_rnn_gluon(model, num_hiddens, corpus_indices, vocab,
                                ctx, num_epochs, num_steps, lr,
                                clipping_theta, batch_size, prefixes):
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})
    start = time.time()
    for epoch in range(num_epochs):
        l_sum, n = 0.0, 0
        data_iter = d2l.data_iter_consecutive(
            corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for X, Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            # Clip the gradient
            grad_clipping_gluon(model, clipping_theta, ctx)
            # Since the error has already taken the mean, the gradient does
            # not need to be averaged
            trainer.step(1)
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % 50 == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            start = time.time()
        if (epoch + 1) % 100 == 0:
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(prefix, 50, model, vocab, ctx))

num_epochs, batch_size, lr, clipping_theta = 500, 32, 1, 1
pred_period, pred_len, prefixes = 50, 50, ['traveller', 'time traveller']
train_and_predict_rnn_gluon(model, num_hiddens, corpus_indices, vocab, ctx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, prefixes)
