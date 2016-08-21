#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""テキスト用のツール群
"""
from requests_oauthlib import OAuth1Session
#from BeautifulSoup import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import json
import urllib
import copy
import pickle
import MeCab
import random
import pickle
import os
import CCAA
import clone
import numpy as np
from twitter_tools import Twitter


#MECAB_MODE = 'mecabrc'
PARSE_TEXT_ENCODING = 'utf-8'

stop_list=(u"RT",u"http")

#PARSE_TEXT_ENCODING = 'utf-8'
#MECAB_MODE = 'mecabrc'
PARSE_TEXT_ENCODING = 'utf-8'
#MECAB_MODE = '-Ochasen -d /usr/local/Cellar/mecab/0.996/lib/mecab/dic/mecab-ipadic-neologd/'

stop_list=(u"RT",u"http")

class TextTools:
    MECAB_MODE=" -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"
    tagger = MeCab.Tagger(MECAB_MODE)
    tagger.parse("")
    @classmethod
    def parse(cls, text):
        u"""記号を取り除きスペース区切りでテキストを返す
        入力：str
        返値：str
        """
        node = cls.tagger.parseToNode(text).next
        result=""
        while node:
            word=node.surface
            print(word,word.isalpha())
            #if word.isalpha() or "/" in word or "%" in word or "[]" in word or "「" in word or "」" in word:
            #    node = node.next
            #    continue
            word = node.surface
            node = node.next
            result+=word+" "
        return result

    @classmethod
    def conect_timeline(cls, timeline,rep=1):
        """ツイートのリストを一つの文字列にして返す
        """
        return_text=""
        for tweet in timeline:
            if "http" in tweet["text"]:
                continue
            if not rep:
                a = tweet["text"].rsplit()
                tweet[0]=""
                for i in a:
                    if i[0] != "@":
                        tweet[0]+=i+u"。"
                tweet[0]=tweet[0][:-1]
                return_text+=tweet[0].replace("\n",u"。")+u"。"
                #return_text=re.sub(re.compile("[a-zA-Z0-9_]"), '', return_text)
        return return_text

def test():
    text = u"高椅くんは、勉強しなかったので、点数がとれず、悔しがっていたのをいま思い出すと、残念だ。"
    return TextTools.parse(text)

def create_train(ids, save=1,rep = 0,path="TimeLine/"):
    """指定idのツイートを取得し保存する
    すでに保存してあるツイートがある場合そこに追加する（ツイートの重複はしない）
    引数：指定id、ツイートを保存するか(デフォルトはする)、リプライを入れるか（デフォルトは入れない）、ディレクトリ(def:TimeLine)
    返り値：ツイートを結合したテキストと使用したapiの回数
    """
    since_id=1
    max_id=None
    path += "TimeLine"+ids
    twitter = Twitter(ids)
    timeLine=[]
    count = 0
    oldLine = []

    if os.path.exists(path):
        oldLine=pickle.load(open(path,"rb"))
        since_id=oldLine[0][1]
    else:
        print ("user id ",ids,"is first")

    for i in range(40):
        train_twit=clone.get_twit_list(ids, rep=rep, max_id=max_id, since_id=since_id)
        if len(train_twit)<=0:
            break
        for j in train_twit:
            if rep:
                if j[0][0] != u"@":
                    continue
            timeLine.append(j)
        count +=len(train_twit)
        max_id=int(train_twit[-1][0])-1

    timeLine += oldLine
    train_data=conect_timeline(timeLine,rep=rep)
    if count != 0 and save:
        pickle.dump(timeLine,open(path,"wb"),-1)

    print ("added ",count," tweets:use api ",i+1)

    return train_data.encode(PARSE_TEXT_ENCODING),i+1

def make_dataset(tweetPath=None,vocabPath="data/vocab.bin"):
    """データ・セットを作る
    引数：対象のタイムラインとそのボキャブラリー
    返値：データセットと
    """
    if not tweetPath:
        tweetPath = "TimeLine/TimeLine"+CCAA.target
    vocab = pickle.load(open(vocabPath,"rb")) if os.path.exists(vocabPath) else {}
    #text=codecs.open(textPath ,'rb', 'UTF-8').read()
    text = conect_timeline(pickle.load(open(tweetPath,"rb")))
    tagger = MeCab.Tagger(MECAB_MODE)
    #text = text.encode(PARSE_TEXT_ENCODING)
    node = tagger.parseToNode(text)
    result=[]

    while node:
        word = node.surface#.decode("utf-8")
        feature = node.feature.split(',')
        node = node.next
        if word:
            if word not in vocab:
                vocab[word] = (len(vocab),feature)
            result.append(word)

    dataset = np.ndarray((len(result),), dtype=np.int32)

    for i, word in enumerate(result):
        dataset[i] = vocab[word][0]
    print ('corpus length:', len(result))
    print ('vocab size:', len(vocab))
    return dataset, result, vocab
