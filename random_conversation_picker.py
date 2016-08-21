import clone
import pickle
import time
import textTools
import os,sys
import random
import CCAA

user_list = "user_list.csv"
save_dir = "conversation_list"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
    target_id = CCAA.target
    name = clone.id_to_name(target_id)["name"]
    with open(os.path.join(save_dir,user_list),"w") as f:
        f.write("{},{}\n".format(target_id,name))

sys.exit()

rand_list=[i.rsplit() for i in open(user_list,"wb")]
exchange_list=[i[8:] for i in os.listdir("TimeLine") if not i[0]=="."]
all_list=set(rand_list+exchange_list)
all_list.remove(CCAA.target)
all_list=list(all_list)
random.shuffle(all_list)
pick=[]
for i in all_list[:10]:
    i = clone.user_info(i)
    if not i is None:
        ids,name = i["id"],i["name"]
        print (name)
        pick+=clone.follow_exchanger(ids)
pick=[i for i in pick if not i in all_list]
random.shuffle(pick)
count=0
start_time=-1
for i in pick[:30]:
    ids=i
    print ("get ",ids ,"'s tweet")
    if start_time == -1:
        count=0
        start_time =time.time()
    words,num=textTools.create_train(ids,path="random_TimeLine/")
    count+=num

    if count >= 150:
        sleep_time = 900 - (time.time()-start_time) if 900 - (time.time()-start_time) > 0 else 0
        print ("limit!! sleep : ",sleep_time,"s")
        time.sleep(sleep_time)
        start_time=-1
