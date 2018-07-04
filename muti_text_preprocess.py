# coding: utf-8

import json
import re
import time
import jieba
import jieba.analyse
import numpy as np
import multiprocessing as mp 
import pickle
jieba.set_dictionary('jieba_dict/dict.txt.big')
jieba.initialize()

def word_segmentation(dicts):
    text = filter1.sub('', dicts['article'])
    text = filter2.sub('', text)
    return {"article" : " ".join(jieba.analyse.textrank(text , topK=100 , withWeight=False)) , "site_category" : dicts['site_category']}

if __name__ == "__main__":
    
    lines = []
    
    with open('5_nomove.jl', 'r' , encoding='utf-8') as file:
        for line in file:
            lines.append(json.loads(line))

    #只取pixnet中title跟body當作training data ， site_category當作testing data
    data = [{"article":i["title"]+i["body"] , "site_category":i["site_category"]} for i in lines]
    
    filter1 = re.compile(r'<.*?>|&lt;|&gt;|&amp;|&nbsp;|&quot;') #過濾html的tag名稱
    filter2 = re.compile(r'\W+') #過濾特別的char


    steps = int(len(data)/1000)
    token_data = []
    
    #以1000筆為一份跑mutiprocess 不然記憶體裝不下
    for step in range(steps+1):
        start_time = time.time()
        pool = mp.Pool()
        if step == steps:
            result = pool.map(word_segmentation, data[ step*1000 : ])
        else:
            result = pool.map(word_segmentation, data[ step*1000 : step*1000+999 ])
        pool.close()
        for r in result:
            token_data.append(r)
        print("step:%d    ,    time:%f"%( step*1000 , time.time()-start_time))
    

    with open('token_list.pickle', 'wb') as handle:
        pickle.dump(token_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

