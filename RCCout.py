#%%
# -*- coding: utf-8 -*-
import time
import math
import sys
import argparse
import pickle
import random

import numpy as np
from chainer import cuda, Variable, FunctionSet
import chainer.functions as F

from CharRNN import CharRNN#, make_initial_state
from tools import Twitter

class Clone:
    def __init__(self,target_ids):
        self.twitter = Twitter(target_ids)
        self.train_count=0

    def train(self):
        #while True:
        print ("Start {} times learning.".format(self.train_count))
        exec(open("train.py").read())
        print ("{} times learning done.".format(self.train_count))
        self.train_count+=1


    def tweet(self):
        if time.localtime()[3] > 6:
            a,b=textTools.create_train(clone.target,save=1)
            with open("data/Target.txt","w") as f:
                f.write(a)
        twit=RCCout.Speak()
        twit=twit.rsplit(u"。")
        tNum=np.random.randint(2,4)
        tList=random.sample(range(len(twit)),tNum)
        ttt=[twit[i] for i in tList]
        tweet=ttt[0]

        try:
            clone.twit_post(tweet)
        except ReadTimeout as e:
            print ("ReadTimeout")
            print ("waiting 5 mins")
            time.sleep(5*60)
        except ConnectionError as e:
            print ("ConnectionError")
            print ("waiting 5mins")
            time.sleep(5*60)


def Speak(model="data/model",vocabulary="data/vocab.bin",seed=1,sample=1,length=2000,gpu=-1):
    #np.random.seed(seed)
    # load vocabulary
    vocab = pickle.load(open(vocabulary, 'rb'))
    ivocab = {}
    for c, i in vocab.items():
        ivocab[i[0]] = c
    model = pickle.load(open(model, 'rb'))
    n_units = model.embed.W.data.shape[1]

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # initialize generator
    state = make_initial_state(n_units, batchsize=1, train=False)
    if gpu >= 0:
        for key, value in state.items():
            value.data = cuda.to_gpu(value.data)

    prev_char = np.array([0], dtype=np.int32)
    if gpu >= 0:
        prev_char = cuda.to_gpu(prev_char)
    primetext=random.sample(vocab.keys(),1)
    if len(primetext) > 0:
        for i in primetext:
            sys.stdout.write(i)
            prev_char = np.ones((1,), dtype=np.int32) * vocab[i][0]
            if gpu >= 0:
                prev_char = cuda.to_gpu(prev_char)
            state, prob = model.forward_one_step(prev_char,prev_char,state,train=False)
            #state, prob = model.predict(prev_char, state)
    twit=""
    for i in xrange(length):
        state, prob = model.forward_one_step(prev_char,prev_char,state,train=False)

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

        if gpu >= 0:
            prev_char = cuda.to_gpu(prev_char)
    return twit
