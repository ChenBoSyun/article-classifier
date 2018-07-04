# coding: utf-8
#載入模型做預測 將預測結果顯示出來


import keras
from keras.models import load_model
from data import Data_preprocessor
import numpy as np
import pickle



def predict(article):
    X = np.array(article)
    X = X.reshape(1,100)
    answer = model.predict_classes(X)
    result = index['int_category'][answer[0]]
    print(result)
    return result



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--file',
                       default='test.txt',
                       help='input test data file name')
    args = parser.parse_args()
    
    with open(args.file , 'r' , encoding='utf-8') as file:     
        article = file.read()
        
    model = load_model('classifier.h5')
    index = pickle.load(open("index.pickle","rb"))
    
    preprocessor = Data_preprocessor(article)
    article = preprocessor.preprocess()
    
    result = predict(article)
