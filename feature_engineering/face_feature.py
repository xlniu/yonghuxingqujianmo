#encoding:utf-8
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

#读入人脸特征
train_face = pd.read_csv('./train/train_face.txt',header=None,sep = '\t',names = ['photo_id','face_property'])
test_face = pd.read_csv('./test/test_face.txt',header=None,sep = '\t',names = ['photo_id','face_property'])
data = pd.concat([train_face,test_face])

def face_property(x):
    x = x[2:len(x)-2]
    x = x.split('], [')
    new_x = []
    for i in x:
        new_x.append(i.split(','))
    new_x = np.array(new_x)
    pro = new_x[:,0]
    sex = new_x[:,1]
    age = new_x[:,2]
    look = new_x[:,3]
    
    return [pro,sex,age,look]
data['property_list'] = data['face_property'].apply(face_property)
data['pro'] = data['property_list'].apply(lambda x : x[0])
data['sex'] = data['property_list'].apply(lambda x : x[1])
data['age'] = data['property_list'].apply(lambda x : x[2])
data['look'] = data['property_list'].apply(lambda x : x[3])

def pro(x):
    result = np.zeros(7)
    x = np.array(x).astype(np.float32)
    for i in x:
        if i<0.1:
            result[0]+=1
        elif 0.1<=i<0.2:
            result[1]+=1
        elif 0.2<=i<0.3:
            result[2]+=1
        elif 0.3<=i<0.4:
            result[3]+=1
        elif 0.4<=i<0.5:
            result[4]+=1
        elif 0.5<=i<0.6:
            result[5]+=1
        else:
            result[6]+=1
    return result
data['pro_list'] = data['pro'].apply(pro)

def sex(x):
    result = np.zeros(2)
    x = np.array(x).astype(int)
    for i in x:
        if i==0:
            result[0]+=1
        else:
            result[1]+=1
    return result
data['sex_list'] = data['sex'].apply(sex)

def age(x):
    result = np.zeros(20)
    x = np.array(x).astype(int)
    for i in x:
        if i<2:
            result[0]+=1
        elif 2<=i<3:
            result[1]+=1
        elif 3<=i<5:
            result[2]+=1
        elif 5<=i<6:
            result[3]+=1
        elif 6<=i<8:
            result[4]+=1
        elif 8<=i<10:
            result[5]+=1
        elif 10<=i<12:
            result[6]+=1
        elif 12<=i<13:
            result[7]+=1
        elif 13<=i<15:
            result[8]+=1
        elif 15<=i<16:
            result[9]+=1
        elif 16<=i<17:
            result[10]+=1
        elif 17<=i<20:
            result[11]+=1
        elif 20<=i<22:
            result[12]+=1
        elif 22<=i<25:
            result[13]+=1
        elif 25<=i<27:
            result[14]+=1
        elif 27<=i<30:
            result[15]+=1
        elif 30<=i<33:
            result[16]+=1
        elif 33<=i<35:
            result[17]+=1
        elif 35<=i<40:
            result[18]+=1
        else:
            result[19]+=1
    return result
data['age_list'] = data['age'].apply(age)

def look(x):
    result = np.zeros(6)
    x = np.array(x).astype(int)
    for i in x:
        if i<36:
            result[0]+=1
        elif 36<=i<45:
            result[1]+=1
        elif 45<=i<60:
            result[2]+=1
        elif 60<=i<75:
            result[3]+=1
        elif 75<=i<85:
            result[4]+=1
        else:
            result[5]+=1
    return result
data['look_list'] = data['look'].apply(look)

def join_list(a,b,c,d):
    return str(np.hstack((a,b,c,d)))
temp = data.apply(lambda row: join_list(row['pro_list'], row['sex_list'],row['age_list'], row['look_list']), axis=1)
data['face_feature'] = temp

train_columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
train_interaction = pd.read_table('./train/train_interaction.txt',header=None)
train_interaction.columns = train_columns
test_columns = ['user_id','photo_id','time','duration_time']
test_interaction = pd.read_table('./test/test_interaction.txt',header=None)
test_interaction.columns = test_columns
data_interaction = pd.concat([train_interaction,test_interaction])

le_user = LabelEncoder()
data_interaction['user_id'] = le_user.fit_transform(data_interaction['user_id'])
le_photo = LabelEncoder()
data_interaction['photo_id'] = le_photo.fit_transform(data_interaction['photo_id'])

data['photo_id'] = le_photo.transform(data['photo_id'])
data['face_feature'] = data['face_feature'].apply(lambda x:x.replace('\n',''))

def str2list(x):
    x = x[1:-1]
    x = x.split()
    return x
photo_id = data_interaction['photo_id'].unique()
photo_id = pd.DataFrame(photo_id)
photo_id.columns = ['photo_id']
photo_id = pd.merge(photo_id,data,on='photo_id',how='left')
nan_value = '[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]'
photo_id['face_feature'] = photo_id['face_feature'].fillna(nan_value)
photo_id['face_feature'] = photo_id['face_feature'].apply(str2list)
photo_id.sort_values('photo_id',inplace=True)
result = []
for i in photo_id['face_feature'].values:
    result.append(i)

#标准化
result = np.array(result)
scaler = MinMaxScaler()
result = scaler.fit_transform(result)

#保存文件
f = open('face35.pkl','w')
pickle.dump(result,f)
f.close()