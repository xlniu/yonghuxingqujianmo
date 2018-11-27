import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

train_columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
train_interaction = pd.read_table('./train/train_interaction.txt',header=None)
train_interaction.columns = train_columns
test_columns = ['user_id','photo_id','time','duration_time']
test_interaction = pd.read_table('./test/test_interaction.txt',header=None)
test_interaction.columns = test_columns
data = pd.concat([train_interaction,test_interaction])

train_text = pd.read_csv('./train/train_text.txt',header=None,sep='\t',names=['photo_id','description'])
test_text = pd.read_csv('./test/test_text.txt',header=None,sep='\t',names=['photo_id','description']) 
data_text = pd.concat([train_text,test_text])
#将没有文字描述的替换为空串
data_text['description'] = data_text['description'].replace('0','')
data_text['description'] = data_text['description'].apply(lambda x: x.replace(',',' '))

data = pd.merge(data,data_text,on='photo_id',how='left')

le_user = LabelEncoder()
data['user_id'] = le_user.fit_transform(data['user_id'])
le_photo = LabelEncoder()
data['photo_id'] = le_photo.fit_transform(data['photo_id'])
data_text['photo_id'] = le_photo.transform(data_text['photo_id'])

#提取停用词
res = data.groupby('user_id')['description'].apply((lambda x :'&'.join(x))).reset_index()

def word_fre(x):
    word_dict = []
    x = x.split('&')
    docs = []
    for doc in x:
        doc = doc.split()
        docs.append(doc)
        word_dict.extend(doc)
    word_dict = Counter(word_dict)
    new_word_dict = {}
    for key,value in word_dict.items():
        new_word_dict[key] = [value,0]
    del word_dict  
    del x
    for doc in docs:
        doc = Counter(doc)
        for word in doc.keys():
            new_word_dict[word][1] += 1
    return new_word_dict   
res['word_fre'] = res['description'].apply(word_fre)

def top_100(word_dict):
     return sorted(word_dict.items(),key = lambda x:(x[1][1],x[1][0]),reverse = True)[:100]
res['top_100'] = res['word_fre'].apply(top_100)

def top_100_word(word_list):
    words = []
    for i in word_list:
        i = list(i)
        words.append(i[0])
    return words  
res['top_100_word'] = res['top_100'].apply(top_100_word)

word_list = []
for i in res['top_100_word'].values:
    word_list.extend(i)

word_list = Counter(word_list)
word_list = sorted(word_list.items(),key = lambda x:x[1],reverse = True)
user_fre = []
for i in word_list:
    i = list(i)
    user_fre.append(i[1]/data['user_id'].nunique())

stop_words = []
for i,j in zip(word_list,user_fre):
    if j>0.5:
        i = list(i)
        stop_words.append(i[0])

#去除停用词
data_text['description'] = data_text['description'].apply(lambda x: x.split(' '))
data_text['description'] = data_text['description'].apply(lambda line: [w for w in line if w not in stop_words])
data_text['description'] = data_text['description'].apply(lambda x: ' '.join(x))

data_text.sort_values('photo_id',inplace=True)

#计算tfidf特征
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
tfidf = tfidf_vectorizer.fit_transform(data_text['description'].values)
#使用nmf算法，提取文本的主题分布
text_nmf = NMF(n_components=20).fit_transform(tfidf)

#保存文件
f = open('./data/text_nmf20.pkl','w')
pickle.dump(text_nmf,f)
f.close()