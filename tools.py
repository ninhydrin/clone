# -*- coding: utf-8 -*-
"""Twitterや文字列用のクラス
"""
import os
from requests_oauthlib import OAuth1Session
from requests.exceptions import ConnectionError, ReadTimeout, SSLError
import pickle
import json
import urllib
import random

import MeCab
import CCAA
import numpy as np

import CCAA


twit_url = "https://api.twitter.com/1.1/statuses/update.json"
home_url = "https://api.twitter.com/1.1/statuses/home_timeline.json"
url = "https://api.twitter.com/1.1/frends/ids=%s.json"
myFollower = "https://api.twitter.com/1.1/friends/list.json?count=200"
statuses = "https://api.twitter.com/1.1/statuses/user_timeline.json?"
favorites = "https://api.twitter.com/1.1/favorites/"
friends = "https://api.twitter.com/1.1/friends/ids.json?"
users = "https://api.twitter.com/1.1/users/"

target = CCAA.target
stop_list=("RT","http")


class Twitter:
    twitter_oauth = OAuth1Session(CCAA.CK, CCAA.CS, CCAA.AT, CCAA.AS)

    def __init__(self, target_id):
        self.target_id = target_id

    @classmethod
    def __get_method(cls, url):
        try:
            req = cls.twitter_oauth.get(url)
        except ConnectionError as e:
            print ("ConnectionError ",e)
            return None
        except ReadTimeout as e:
            print ("ReadTimeout ",e)
            return None
        if req.status_code != 200:
            print ("miss!!"+str(req.status_code))
            return None
        req = json.loads(req.text)
        return req

    def user_info(self, entity=False):
        url = users+"show.json?user_id={}&include_entities={}".format(self.target_id, entity)
        req = self.__get_method(url)
        return req

    def get_follow(self, toSt=True,count=5000):
        u"""フォローを取得
        """
        url = friends+"user_id={}&stringify_ids={}&count={}".format(self.target_id, toSt, count)
        req = self.__get_method(url)
        return req

    def get_follower(self, toSt=True,count=5000):
        u"""フォロワーを取得
        """
        url = "https://api.twitter.com/1.1/followers/ids.json?user_id={}&stringify_ids={}&count={}"
        url = url.format(self.target_id, toSt, count)
        req = self.__get_method(url)
        return req

    def follow_exchanger(self):
        u"""相互フォロワーを取得
        """
        flw=self.get_follow()
        flwer=self.get_follower()
        if flw is None or flwer is None:
            return []
        return list(set(flw["ids"]).intersection(set(flwer["ids"])))

    def get_twit_list(self, rep=0, rt=0, count=200, max_id=None, since_id=None):
        """指定idのツイート一覧を返す
        引数：指定id、リプライを入れるか（デフォルトは入れない）、リツイートを入れるか（デフォルトは入れない）
        返り値：ツイート一覧のリスト
        """
        url = statuses + "user_id={}&count={}&exclude_replies={}&include_rts={}".format(self.target_id, count, rep^1, rt)
        if max_id:
            url +="&max_id={}".format(max_id) #max_id以下を取得
        if since_id:
            url +="&since_id={}".format(since_id) #since_idより上を取得
        req=self.__get_method(url)
        if req is None:
                return []
        return [{x:twit[x] for x in twit if x=="text"or x=="id_str" or x=="created_at"}for twit in req]

    def create_train_list(self, save=1,rep = 0,path="TimeLine/"):
        """ツイートを取得し保存する
        すでに保存してあるツイートがある場合そこに追加する（ツイートの重複はしない）
        引数：指定id、ツイートを保存するか(def:yes)、リプライを入れるか（def:no）、ディレクトリ(def:TimeLine)
        返り値：ツイートを結合したテキストと使用したapiの回数
        """
        max_id=None
        path += "TimeLine"+self.ids
        twit_count = 0
        api_use_count = 0
        old_timeline = []
        timeline = []

        if os.path.exists(path):
            oldLine=pickle.load(open(path,"rb"))
            since_id=oldLine[0][1]
        else:
            print ("user id ",ids,"is first")

        for i in range(40):
            api_use_count += 1
            new_timeline+=self.get_twit_list(rep=rep, max_id=max_id)
            if len(train_twit)<=0:
                break
            if api_use_count:
                new_timeline = new_timeline[1:] #max_id以下を取得するので一つかぶる
            twit_count +=len(new_timeline)
            max_id=new_timeline[-1]["id_str"]#max_idは馬鹿でかい
            timeline+=new_timeline
        timeline += old_timeline
        train_data=conect_timeline(timeLine, rep=rep)
        if twit_count and save:
            pickle.dump(timeLine,open(path,"wb"),-1)
        print ("added:{} tweet  api: {} used".format(twit_count, api_use_count))
        return (train_data, api_use_count)

    @classmethod
    def word_search(cls, text):
        """特定の言葉を検索する
        """
        url = "https://api.twitter.com/1.1/search/tweets.json?q={}".format(text)
        req=cls.__get_method(url)
        return req

    @classmethod
    def twit_id(cls, ids):
        """指定idのツイートを取得"""
        url = "https://api.twitter.com/1.1/statuses/show.json?id={}".format(ids)
        req=cls.__get_method(url)
        return req

    @classmethod
    def __user_info(cls, ids):
        """指定idのユーザー情報を取得"""
        url = users+"show.json?user_id={}&include_entities={}".format(cls.target_id, entity)
        req=cls.__get_method(url)
        return req

    @classmethod
    def twit_post(cls, content, ids=None):
        url = twit_url
        params = {"status": content}
        if ids:
            url+="?in_reply_to_status_id="+ids
        req = cls.twitter_oauth.post(url, params=params)
        if req.status_code==200:
            return req
        return None


class TextTools:
    #MECAB_MODE=" -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"
    MECAB_MODE = '-d /usr/local/Cellar/mecab/0.996/lib/mecab/dic/mecab-ipadic-neologd/'
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
    def conect_timeline(cls, timeline):
        """ツイートのリストを一つの文字列にして返す
        """
        format_text = ""
        for tweet in timeline:
            if "http" in tweet["text"]:
                continue
            split_text = tweet["text"].rsplit()
            for sentence in split_text:
                if sentence[0] != "@":
                    format_text += sentence + "。"
        return format_text
    @classmethod
    def make_dataset(cls, tweetPath=None, vocabPath="data/vocab.bin"):
        """データ・セットを作る
        引数：対象のタイムラインとそのボキャブラリー
        返値：データセットと
        """
        if not tweetPath:
            tweetPath = "TimeLine/TimeLine"+CCAA.target
        vocab = pickle.load(open(vocabPath,"rb")) if os.path.exists(vocabPath) else {}
        text = cls.conect_timeline(pickle.load(open(tweetPath, "rb")))
        node = cls.tagger.parseToNode(text)
        result=[]

        while node:
            word = node.surface
            feature = node.feature.split(',')
            node = node.next
            if word:
                if not word in vocab:
                    vocab[word] = (len(vocab),feature)
                result.append(word)

        dataset = np.ndarray([vocab[word][0] for word in result], dtype=np.int32)
        print ('corpus length:{}  vocab size:{}'.format(len(result)), len(vocab))
        return (dataset, result, vocab)

def test():
    text = u"高椅くんは、勉強しなかったので、点数がとれず、悔しがっていたのをいま思い出すと、残念だ。"
    return TextTools.parse(text)
