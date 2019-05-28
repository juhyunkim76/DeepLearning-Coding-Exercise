# This source is originally from https://github.com/d2l-ai/d2l-en

import sys
sys.path.insert(0, '..')

from mxnet import nd
import random
import collections

with open('../data/timemachine.txt', 'r') as f:
    raw_text = f.read()
print(raw_text[0:110])

lines = raw_text.split('\n')
text = ' '.join(' '.join(lines).lower().split())
print('# of chars:', len(text))
print(text[0:70])


class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        # sort by frequency and token
        counter = collections.Counter(tokens)
#        print("counter: ", counter)
        token_freqs = sorted(counter.items(), key=lambda x: x[0])
#        print("token_freqs: ", token_freqs)
        token_freqs.sort(key=lambda x : x[1], reverse=True)
#        print("token_freqs: ", token_freqs)

        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        else:
            self.unk = 0
            tokens = ['<unk>']

        tokens += [token for token, freq in token_freqs if freq >= min_freq]
        self.idx_to_token = []
        self.token_to_idx = dict()
        for token in tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        else:
            return [self.__getitem__(tokens) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        else:
            return [self.idx_to_token[index] for index in indices]


vocab = Vocab(text)
print("vocab.token_to_idx: ", vocab.token_to_idx)


corpus_indices = [vocab[char] for char in text]
sample = corpus_indices[:15]

print('chars:', [vocab.idx_to_token[idx] for idx in sample])
print('indices:', sample)


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # Offset for the iterator over the data for uniform starts
    offset = int(random.uniform(0, num_steps))
    corpus_indices = corpus_indices[offset:]

    # Subtract 1 extra since we need to account for the sequence length
    num_examples = ((len(corpus_indices) - 1) // num_steps) -1
    # Discard half empty batches
    num_batches = num_examples // batch_size
    example_indices = list(range(0, num_examples * num_steps, num_steps))
    random.shuffle(example_indices)

    # This returns a sequence of the length num_steps starting from pos
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(0, batch_size * num_batches, batch_size):
        # Batch_size indicates the random examples read each time
        batch_indices = example_indices[i:(i+batch_size)]
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)


my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    # Offset for the iterator over the data for uniform starts
    offset = int(random.uniform(0,num_steps))
    # Slice out data - ignore num_steps and just wrap around
    num_indices = ((len(corpus_indices) - offset) // batch_size) * batch_size
    indices = nd.array(corpus_indices[offset:(offset + num_indices)], ctx=ctx)
    indices = indices.reshape((batch_size, -1))
#    print("indices: ", indices)
    # Need to leave one last token since targets are shifted by 1
    num_epochs = ((num_indices // batch_size)  - 1) // num_steps
    for i in range(0, num_epochs * num_steps, num_steps):
        X = indices[:,i:(i+num_steps)]
        Y = indices[:,(i+1):(i+1+num_steps)]
        yield X, Y

for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y)
