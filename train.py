import codecs
import time
import math
import sys
import argparse
import pickle
import copy
import os

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import MeCab
from tools import Twitter, TextTools

from CharRNN import CharRNN, make_initial_state

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu',                        type=int,   default=-1)
parser.add_argument('--rnn_size',                   type=int,   default=128)
parser.add_argument('--learning_rate',              type=float, default=2e-3)
parser.add_argument('--learning_rate_decay',        type=float, default=0.97)
parser.add_argument('--learning_rate_decay_after',  type=int,   default=10)
parser.add_argument('--decay_rate',                 type=float, default=0.95)
parser.add_argument('--dropout',                    type=float, default=0.0)
parser.add_argument('--seq_length',                 type=int,   default=50)
parser.add_argument('--batchsize',                  type=int,   default=50)
parser.add_argument('--epochs',                     type=int,   default=5)
parser.add_argument('--grad_clip',                  type=int,   default=5)
parser.add_argument('--init_from',                  type=str,   default='/model')

args = parser.parse_args()

n_epochs    = args.epochs
n_units     = args.rnn_size
batchsize   = args.batchsize
bprop_len   = args.seq_length
grad_clip   = args.grad_clip

#tweetPath='TimeLine/TimeLine'+CCAA.target
vocabPath='data/vocab.bin'
modelPath="data/model"

train_data, words, vocab = TextTools.make_dataset()
pickle.dump(vocab, open(vocabPath, 'wb'))

if os.path.exists(modelPath):
    model = pickle.load(open(modelPath, 'rb'))
else:
    model = CharRNN(len(vocab), n_units)

count = 0
while len(model.l3.b) < len(vocab):
    model.add_unit()
    count+=1
if count:
    print (count,"units added")
del words, vocab

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

optimizer = optimizers.RMSprop(lr=args.learning_rate, alpha=args.decay_rate, eps=1e-8)
optimizer.setup(model)

whole_len    = train_data.shape[0]
jump         = whole_len / batchsize
epoch        = 0
start_at     = time.time()
cur_at       = start_at
state        = make_initial_state(n_units, batchsize=batchsize)


if args.gpu >= 0:
    accum_loss   = Variable(cuda.zeros(()))
    for key, value in state.items():
        value.data = cuda.to_gpu(value.data)#plist
else:
    accum_loss   = Variable(np.zeros((), dtype=np.float32))


print ('going to train {} iterations'.format(jump * n_epochs))

for i in xrange(jump * n_epochs):
    x_batch = np.array([train_data[(jump * j + i) % whole_len]
                        for j in xrange(batchsize)])
    y_batch = np.array([train_data[(jump * j + i + 1) % whole_len]
                        for j in xrange(batchsize)])

    if args.gpu >=0:
        x_batch = cuda.to_gpu(x_batch)
        y_batch = cuda.to_gpu(y_batch)

    state, loss_i = model.forward_one_step(x_batch, y_batch, state, dropout_ratio=args.dropout)
    accum_loss   += loss_i

    if (i + 1) % bprop_len == 0:  # Run truncated BPTT
        now = time.time()
        sys.stderr.write('\r{}/{}, train_loss = {}, time = {:.2f}'.format((i+1)/bprop_len,(jump*n_epochs)/bprop_len, accum_loss.data / bprop_len, now-cur_at))
        sys.stderr.flush()
        #print '{}/{}, train_loss = {}, time = {:.2f}'.format((i+1)/bprop_len,(jump*n_epochs)/bprop_len, accum_loss.data / bprop_len, now-cur_at)
        cur_at = now

        optimizer.zero_grads()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        if args.gpu >= 0:

            accum_loss = Variable(cuda.zeros(()))
        else:
            accum_loss = Variable(np.zeros((), dtype=np.float32))

        optimizer.clip_grads(grad_clip)
        optimizer.update()

    if (i + 1) % 10000 == 0:
        pickle.dump(copy.deepcopy(model).to_cpu(), open(modelPath, 'wb'))

    if (i + 1) % jump == 0:
        epoch += 1

        if epoch >= args.learning_rate_decay_after:
            optimizer.lr *= args.learning_rate_decay
            print ('decayed learning rate by a factor {} to {}'.format(args.learning_rate_decay, optimizer.lr))

    sys.stdout.flush()

pickle.dump(copy.deepcopy(model).to_cpu(), open(modelPath, 'wb'))    


class train:
    def __init__(self, model, model_path):
        self.model = model
        self.model_path = model_path

    def train(self, train_data, **kwargs):

            gpu = -1 if "gpu" not in kwargs else kwargs["gpu"]
            lr = 2e-3 if "lr" not in kwargs else kwargs["lr"]
            lr_decay = 0.97 if "lr_decay" not in kwargs else kwargs["lr_decay"]
            lr_decay_after=10 if "lr_decay_after" not in kwargs else kwargs["lr_decay_after"]
            decay_rate = 0.95 if "decay_rate" not in kwargs else kwargs["decay_rate"]
            dropout = 0.0 if "dropout" not in kwargs else kwargs["dropout"]
            bprop_len   = 50 if "bprop_len" not in kwargs else kwargs["bprop_len"]
            batchsize   = 50 if "batchsize" not in kwargs else kwargs["batchsize"]
            grad_clip   = 5 if "grad_clip" not in kwargs else kwargs["grad_clip"]

            if gpu >= 0:
                cuda.get_device(gpu).use()
                self.model.to_gpu()

            optimizer = optimizers.RMSprop(lr=lr, alpha=decay_rate, eps=1e-8)
            optimizer.setup(self.model)

            whole_len    = train_data.shape[0]
            jump         = whole_len / batchsize
            epoch        = 0
            start_at     = time.time()
            cur_at       = start_at
            state        = self.model.make_initial_state(batchsize=batchsize)

            if gpu >= 0:
                accum_loss   = Variable(cuda.zeros(()))
            for key, value in state.items():
                value.data = cuda.to_gpu(value.data)#plist
            else:
                accum_loss   = Variable(np.zeros((), dtype=np.float32))

            print ('going to train {} iterations'.format(jump * epochs))

            for i in range(jump * n_epochs):
                x_batch = np.array([train_data[(jump * j + i) % whole_len]
                                    for j in range(batchsize)])
                y_batch = np.array([train_data[(jump * j + i + 1) % whole_len]
                                    for j in range(batchsize)])

                if args.gpu >=0:
                    x_batch = cuda.to_gpu(x_batch)
                    y_batch = cuda.to_gpu(y_batch)

                state, loss_i = model.forward_one_step(x_batch, y_batch, state, dropout_ratio=dropout)
                accum_loss   += loss_i

                if (i + 1) % bprop_len == 0:  # Run truncated BPTT
                    now = time.time()
                    sys.stderr.write('\r{}/{}, train_loss = {}, time = {:.2f}'.format((i+1)/bprop_len,(jump*n_epochs)/bprop_len, accum_loss.data / bprop_len, now-cur_at))
                    sys.stderr.flush()
                    cur_at = now

                    optimizer.zero_grads()
                    accum_loss.backward()
                    accum_loss.unchain_backward()  # truncate

                    if gpu >= 0:
                        accum_loss = Variable(cuda.zeros(()))
                    else:
                        accum_loss = Variable(np.zeros((), dtype=np.float32))

                    optimizer.clip_grads(grad_clip)
                    optimizer.update()

                    if (i + 1) % 10000 == 0:
                        pickle.dump(copy.deepcopy(self.model).to_cpu(), open(self.model_path, 'wb'))

                    if (i + 1) % jump == 0:
                        epoch += 1

                    if epoch >= learning_rate_decay_after:
                        optimizer.lr *= learning_rate_decay
                        print ('decayed learning rate by a factor {} to {}'.format(learning_rate_decay, optimizer.lr))
                sys.stdout.flush()

            pickle.dump(copy.deepcopy(self.model).to_cpu(), open(self.model_path, 'wb'))
