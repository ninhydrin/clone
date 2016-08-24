#%%
# -*- coding: utf-8 -*-
import time
import math
import sys
import argparse
import pickle
import random
import copy
import os
from requests.exceptions import ConnectionError, ReadTimeout, SSLError

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

from CharRNN import CharRNN#, make_initial_state
from tools import Twitter, TextTools
import CCAA

class Clone:
    def __init__(self, target_ids, model=None, rnn_size=128, gpu=-1):
        self.twitter = Twitter(target_ids)
        self.train_count=0
        self.gpu = gpu
        self.model_path = "dada/model{}.pkl".format(target_ids)
        self.tweet_path = "TimeLine/TimeLine"+target_ids
        self.vocab_path = "data/vocab{}.bin".format(target_ids)
        self.vocab = pickle.load(open(self.vocab_path,"rb")) if os.path.exists(self.vocab_path) else {}

        if os.path.exists(self.model_path):
            self.model = pickle.load(open(self.model_path, 'rb'))
        else:
            self.model = CharRNN(len(self.vocab), rnn_size)

    def train_loop(self, num=100):
        for i in range(num):
            if (i % 5 ) == 0:
                self.make_dataset()
            self.train()
            self.tweet()

    def train(self, **kwargs):
        print ("Start {} times learning.".format(self.train_count))
        self._train(*kwargs)
        print ("{} times learning done.".format(self.train_count))
        self.train_count+=1

    def make_dataset(self):
        self.twitter.save_timeline()
        dataset, result, vocab = TextTools.make_dataset(self.tweet_path, self.vocab)
        self.dataset = dataset
        self.result = result
        self.vocab = vocab

        count = 0
        pickle.dump(self.vocab, open(self.vocab_path, 'wb'))
        while len(self.model.l3.b) < len(self.vocab):
            self.model.add_unit()
            count+=1
        if count:
            print (count,"units added")

    def tweet(self):
        if time.localtime()[3] > 6:
            pass
        twit=self.Speak()
        twit=twit.rsplit(u"。")
        tNum=np.random.randint(1, min(4, len(twit)))
        tList=random.sample( range(len(twit)), tNum)
        ttt=[twit[i] for i in tList]
        tweet=ttt[0]
        try:
            Twitter.twit_post(tweet)
        except ReadTimeout as e:
            print ("ReadTimeout")
            print ("waiting 5 mins")
            time.sleep(5*60)
        except ConnectionError as e:
            print ("ConnectionError")
            print ("waiting 5mins")
            time.sleep(5*60)

    def Speak(self, seed=1, sample=1, length=200):
        ivocab = {}
        for c, i in self.vocab.items():
            ivocab[i[0]] = c
        n_units = self.model.embed.W.data.shape[1]

        if self.gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu()

        # initialize generator
        state = self.model.make_initial_state(batchsize=1, train=False)
        if self.gpu >= 0:
            for key, value in state.items():
                value.data = cuda.to_gpu(value.data)

        prev_char = np.array([0], dtype=np.int32)
        if self.gpu >= 0:
            prev_char = cuda.to_gpu(prev_char)
        primetext=random.sample(self.vocab.keys(),1)
        if len(primetext) > 0:
            sys.stdout.write(primetext[0])
            prev_char = np.array([self.vocab[primetext[0]][0]], dtype=np.int32)
            if self.gpu >= 0:
                prev_char = cuda.to_gpu(prev_char)
        twit=""
        for i in range(length):
            state, prob = self.model.forward_one_step(prev_char,prev_char,state,train=False)
            if sample > 0:
                probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
                probability /= np.sum(probability)
                index = np.random.choice(range(len(probability)), p=probability)
            else:
                index = np.argmax(cuda.to_cpu(prob.data))
                prob.data[index]=0
                index2 = np.argmax(cuda.to_cpu(prob.data))
                prob.data[index2]=0
                index3 = np.argmax(cuda.to_cpu(prob.data))
                prob.data[index3]=0
                index = random.sample([index,index2,index3],1)[0]

            if index != 0:
                twit+=ivocab[index]
            prev_char = np.array([index], dtype=np.int32)
            if self.gpu >= 0:
                prev_char = cuda.to_gpu(prev_char)
        return twit

    def _train(self, **kwargs):
            gpu = -1 if "gpu" not in kwargs else kwargs["gpu"]
            lr = 2e-3 if "lr" not in kwargs else kwargs["lr"]
            lr_decay = 0.97 if "lr_decay" not in kwargs else kwargs["lr_decay"]
            lr_decay_after=10 if "lr_decay_after" not in kwargs else kwargs["lr_decay_after"]
            decay_rate = 0.95 if "decay_rate" not in kwargs else kwargs["decay_rate"]
            dropout = 0.0 if "dropout" not in kwargs else kwargs["dropout"]
            bprop_len   = 50 if "bprop_len" not in kwargs else kwargs["bprop_len"]
            batchsize   = 50 if "batchsize" not in kwargs else kwargs["batchsize"]
            grad_clip   = 5 if "grad_clip" not in kwargs else kwargs["grad_clip"]
            n_epochs  = 5 if "epochs" not in kwargs else kwargs["epochs"]
            if gpu >= 0:
                cuda.get_device(gpu).use()
                self.model.to_gpu()

            optimizer = optimizers.RMSprop(lr=lr, alpha=decay_rate, eps=1e-8)
            optimizer.setup(self.model)

            train_data = self.dataset
            whole_len    = train_data.shape[0]
            jump         = whole_len // batchsize
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

            print ('going to train {} iterations'.format(jump * n_epochs))

            for i in range(jump * n_epochs):
                x_batch = np.array([train_data[(jump * j + i) % whole_len]
                                    for j in range(batchsize)])
                y_batch = np.array([train_data[(jump * j + i + 1) % whole_len]
                                    for j in range(batchsize)])

                if gpu >=0:
                    x_batch = cuda.to_gpu(x_batch)
                    y_batch = cuda.to_gpu(y_batch)

                state, loss_i = self.model.forward_one_step(x_batch, y_batch, state, dropout_ratio=dropout)
                accum_loss   += loss_i

                if (i + 1) % bprop_len == 0:  # Run truncated BPTT
                    now = time.time()
                    sys.stderr.write('\r{}/{}, train_loss = {}, time = {:.2f}'.format((i+1)//bprop_len,(jump*n_epochs)//bprop_len, accum_loss.data / bprop_len, now-cur_at))
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

                    if epoch >= lr_decay_after:
                        optimizer.lr *= lr_decay
                        print ('decayed learning rate by a factor {} to {}'.format(lr_decay, optimizer.lr))
                sys.stdout.flush()

            pickle.dump(copy.deepcopy(self.model).to_cpu(), open(self.model_path, 'wb'))
