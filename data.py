
# coding: utf-8

# In[2]:


import re
import jieba
import jieba.analyse
import pickle
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


class Data_preprocessor():
    def __init__(self , article):
        jieba.set_dictionary('jieba_dict/dict.txt.big')
        jieba.initialize()
        
        self.filter1 = re.compile(r'<.*?>|&lt;|&gt;|&amp;|&nbsp;|&quot;') #過濾html的tag名稱
        self.filter2 = re.compile(r'\W+') #過濾特別的char
        
        self.index = pickle.load(open("index.pickle","rb"))
        self.article = article 
    
    def preprocess(self):
        self.word_segmentation()
        return self.to_oneHot()
        
    def to_oneHot(self):
        X = []
        for word in self.article.split():
            try:
                X.append(self.index['corpus_int'][word])
            except:
                X.append(0)
        for i in range(100-len(X)):
            X.append(0)
        
        return X
        
    def word_segmentation(self):
        text = self.filter1.sub('', self.article)
        text = self.filter2.sub('', text)
        self.article =  " ".join(jieba.analyse.extract_tags(text , topK=100))

