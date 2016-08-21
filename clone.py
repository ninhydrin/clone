#!/usr/bin/env python
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

# ツイート投稿用のURL
twit_url = "https://api.twitter.com/1.1/statuses/update.json"
home_url = "https://api.twitter.com/1.1/statuses/home_timeline.json"
url = "https://api.twitter.com/1.1/frends/ids=%s.json"
myFollower = "https://api.twitter.com/1.1/friends/list.json?count=200"
statuses = "https://api.twitter.com/1.1/statuses/user_timeline.json?"
favorites = "https://api.twitter.com/1.1/favorites/"
friends = "https://api.twitter.com/1.1/friends/ids.json?"
users = "https://api.twitter.com/1.1/users/"

target = CCAA.target
twitter = OAuth1Session(CCAA.CK, CCAA.CS, CCAA.AT, CCAA.AS)
#twitter = OAuth1Session(CK, CS, AT, AS)

def user_info(ids,entity=False):
    url = users+"show.json?user_id={}&include_entities={}".format(ids,entity)
    req = make_info(url)
    return req

def users_info(ids):
    """
    引数：ユーザーid(複数可)
    返値：プロフィール
    """
    count = 0
    url = users+"lookup.json?user_id="
    for i in ids:
        url += i+","
        count+=1
        if count>100:
            break
    req = make_info(url[:-1])
    return req

def get_follow(ids,toSt=True,count=5000):
    u"""指定idのフォローを取得
    """
    url = friends+"user_id={}&stringify_ids={}&count={}".format(ids,toSt,count)
    req = make_info(url)
    return req

def get_follower(ids,toSt=True,count=5000):
    u"""指定idのフォロワーを取得
    """
    url = "https://api.twitter.com/1.1/followers/ids.json?user_id={}&stringify_ids={}&count={}".format(ids,toSt,count)
    req = make_info(url)
    return req


def follow_exchanger(ids):
    u"""指定idの相互フォロワーを取得
    """
    flw=get_follow(ids)
    flwer=get_follower(ids)
    if flw is None or flwer is None:
        return []
    return list(set(flw["ids"]).intersection(set(flwer["ids"])))

#ゲット用
def twit_get(use_url):
    try:
        req = twitter.get(use_url, params = params)
    except ConnectionError as e:
        print ("ConnectionError ",e)
        return None
    except ReadTimeout as e:
        print ("ReadTimeout ",e)
        return None
    if req.status_code==200:
        return req
    print ("miss!!"+str(req.status_code))
    return None

#ポスト用
def twit_post(content,account=twitter,ids=None):
    url = twit_url
    if ids:
        url+="?in_reply_to_status_id="+ids
    req = account.post(url, params = content)
    if req.status_code==200:
        return req
    "miss!!"
    return None

#ツイート一覧のリストを返す
def get_twit_list(ids,rep=0,rt=0,count=200,max_id=None,since_id=None):
    """指定idのツイート一覧を返す
    引数：指定id、リプライを入れるか（デフォルトは入れない）、リツイートを入れるか（デフォルトは入れない）
    返り値：ツイート一覧のリスト
    """
    url = statuses + "user_id={}&count={}&exclude_replies={}&include_rts={}".format(ids,count,rep^1,rt)
    if max_id:
        url +="&max_id={}".format(max_id) #max_id以下を取得
    if since_id:
        url +="&since_id={}".format(since_id) #since_idより上を取得
    req = make_info(url)
    if req is None:
        return []
    return [[twit[x] for x in twit if x=="text"or x=="id_str" or x=="created_at" ]for twit in req]

def max_list(ids,rep=1,rt=0):
    twit_list=[]
    ttid=None
    for i in range(30):
        get_list=get_twit_list(ids,rep=rep,rt=rt,tid=ttid)
        if len(get_list) <= 1:
            break
        ttid=get_list[-1][1]
        twit_list+=get_list
    return twit_list,i+1

def home_timeline():
    """自身のホームタイムラインを返す
    """
    req = twitter.get(home_url, params = {"status":"ok"})
    if req.status_code==200:
        return json.loads(req.text)
    print ("miss!!")
    return None

def make_info(use_url):
    #users_info作成
    req=twit_get(use_url)
    if req is None:
        return None
    req = json.loads(req.text)
    return req

def get_rep(max_id = None,since_id=None):
    #自分へのリプライを取得
    url="https://api.twitter.com/1.1/statuses/mentions_timeline.json?"

    if max_id:
        url +="&max_id={}".format(max_id) #max_id以下を取得
    if since_id:
        url +="&since_id={}".format(since_id) #since_idより上を取得

    req = twitter.get(url, params = {"status":"ok"})
    if req.status_code==200:
        req = json.loads(req.text)
        return [[twit[x] for x in twit if x=="text"or x=="id_str" or x=="created_at" or x=="entities" or x == "user"]for twit in req]
    print ("miss!!")
    return None

#IDからツイートを取得

def id_to_name(ids):
    """指定idのユーザー情報を取得"""
    url = "https://api.twitter.com/1.1/users/show.json?id={}".format(ids)
    req=twit_get(url)
    req = json.loads(req.text)
    return req

#ファボ用
def favo(ids , unfavo = False):
    """ふぁぼる
    """
    url = favorites + "create.json?id={}".format(ids)
    if unfavo:
        url = favorites + "destroy.json?id={}".format(ids)
    twit_post(url)

# ツイート本文
params = {"status": ""}
