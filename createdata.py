# -*- coding: utf-8 -*-
"""指定ユーザーの相互フォロワーのツイートを取得
基本リプライとリツイートは含まない
簡単にbotは除く
"""

import clone
import pickle
import time
import textTools

ids=clone.target

his_list = clone.follow_exchanger(ids)
new_list=[[ids,clone.user_info(ids)["name"]]]
start_time=-1
count=0

print (len(his_list))
while his_list:
    infos=clone.users_info(his_list[:100])
    if start_time == -1:
        count=0
        start_time =time.time()
    count+=1
    for i in infos:
        if not "bot" in i["description"] and not i["protected"]:
            new_list.append([i["id_str"],i["name"]])

    his_list=his_list[100:]
    print "get ",count*100," follower"
    if count == 180:
        sleep_time = 900 - (time.time()-start_time) if 900 - (time.time()-start_time) > 0 else 0
        print "limit!! sleep : ",sleep_time,"s"
        time.sleep(sleep_time)
        count = -1
        start_time=-1

count=0
start_time=-1
for i in new_list:
    ids,name = i
    print "get ",name ,"'s tweet "
    if start_time == -1:
        count=0
        start_time =time.time()
    words,num=textTools.create_train(ids)
    count+=num
    if count >= 150:
        sleep_time = 900 - (time.time()-start_time) if 900 - (time.time()-start_time) > 0 else 0
        print "limit!! sleep : ",sleep_time,"s"
        time.sleep(sleep_time)
        start_time=-1
