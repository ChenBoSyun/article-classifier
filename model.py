
# coding: utf-8

# In[1]:


import pandas as pd

import re
import time
import pickle
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding ,Flatten , Dense
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


data = pickle.load(open("token_list.pickle","rb"))
index = pickle.load(open("index.pickle","rb"))


# In[3]:


y = [index['category_int'][d['site_category']] for d in data]


# In[4]:


y = keras.utils.to_categorical(y)


# In[5]:


X = []
for d in data:
    x = []
    for word in d['article'].split():
        x.append(index['corpus_int'][word])
    X.append(x)


# In[6]:


label_size = len(index['category_int'])
corpus_size = len(index['corpus_int'])
max_length = max([len(i) for i in X])


# In[7]:


X = pad_sequences(X, maxlen=max_length, padding='post')


# In[8]:


model = keras.Sequential()
model.add(Embedding(input_dim = corpus_size, output_dim = 100, input_length=max_length))
model.add(Flatten())
model.add(Dense(label_size , activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model


# In[9]:


# fit the model
model.fit(X, y, nb_epoch=10, batch_size=32, validation_split=0.2, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=1)
print('Accuracy: %f' % (accuracy*100))


# In[ ]:


model.save('classifier.h5')

