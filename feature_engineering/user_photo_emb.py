# coding: utf-8
import commands
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

path = './data/'

train_columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
train_interaction = pd.read_table('./train/train_interaction.txt',header=None)
train_interaction.columns = train_columns
test_columns = ['user_id','photo_id','time','duration_time']
test_interaction = pd.read_table('./test/test_interaction.txt',header=None)
test_interaction.columns = test_columns

data = pd.concat([train_interaction,test_interaction])

le_user = LabelEncoder()
data['user_id'] = le_user.fit_transform(data['user_id'])
le_photo = LabelEncoder()
data['photo_id'] = le_photo.fit_transform(data['photo_id'])

data.sort_values('time',inplace=True)

def generate_doc(df,name,concat_name):
    res = df.astype(str).groupby(name)[concat_name].apply((lambda x :' '.join(x))).reset_index()
    res.columns = [name,'%s_doc'%concat_name]
    return res

user_doc = generate_doc(data,'photo_id','user_id')
photo_doc = generate_doc(data,'user_id','photo_id')

user_doc.to_csv(path+'user_doc.csv',index=False)
photo_doc.to_csv(path+'photo_doc.csv',index=False)

user_doc['user_id_doc'].to_csv(path+'user.adjlist',index=False)
photo_doc['photo_id_doc'].to_csv(path+'photo.adjlist',index=False)

commands.getoutput("bash train_deepwalk.sh")