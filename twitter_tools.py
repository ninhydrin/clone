# -*- coding: utf-8 -*-
"""twitter用の関数群
"""
from requests_oauthlib import OAuth1Session
#from BeautifulSoup import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from requests.exceptions import ConnectionError, ReadTimeout, SSLError
import json
import urllib
import copy
import pickle
import MeCab
import re
import random
import pickle
import os
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

#twitter = OAuth1Session(CK, CS, AT, AS)

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
