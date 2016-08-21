#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import pickle
import MeCab
import random
import time
import clone
import RCCout
import numpy as np
from requests.exceptions import ConnectionError, ReadTimeout, SSLError
counter=0

while True:
    print ("Start ",counter," times learning.")
    #execfile("train.py")
    exec(open("train.py").read())
    print (counter," times learning done.")
    counter+=1

    if time.localtime()[3] > 6:
        if counter%10==0:
            a,b=textTools.create_train(clone.target,save=1)
            fia=open("data/Target.txt","w")
            fia.write(a)
            fia.close()
            counter=0

        twit=RCCout.Speak()
        twit=twit.rsplit(u"。")
        tNum=np.random.randint(2,4)
        tList=random.sample(range(len(twit)),tNum)
        ttt=[twit[i] for i in tList]
        tweet=ttt[0]#+u"。"
        try:
            clone.twit_post({"status":tweet},clone.twitter)
        except ReadTimeout as e:
            #print ("ReadTimeout({0}): {1}".format(errno, strerror))
            print ("ReadTimeout")
            print ("waiting 5 mins")
            time.sleep(5*60)
        except ConnectionError as e:
            #print ("ConnectionError({0}): {1}".format(errno, strerror))
            print ("ConnectionError")
            print ("waiting 5mins")
            time.sleep(5*60)
