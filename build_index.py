# coding: utf-8
#建立index 為了之後做one hot encoding
import pickle

data = pickle.load(open("token_list.pickle","rb"))



int_category = {}
category_int = {}



for i,j in enumerate(set([d['site_category'] for d in data])):
    category_int[j] = i
    int_category[i] = j
    
corpus_int = {}
i=0
for d in data:
    article = d['article'].split()
    for word in article:
        if word not in corpus_int:
            corpus_int[word] = i
            i = i + 1 


index = {'int_category':int_category , 'category_int':category_int , 'corpus_int':corpus_int}

with open('index.pickle', 'wb') as handle:
    pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)

