#encoding:utf-8
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

def read_visual_label(train0_path,train1_path,test_path):
	#读取visual_label
	train0 = pickle.load(open(train0_path,'r'))
	train1 = pickle.load(open(train1_path,'r'))
	test = pickle.load(open(test_path,'r'))

	train_keys = list(train0['visual_data_keys'])+list(train1['visual_data_keys'])
	train_label = list(train0['label'])+list(train1['label'])
	test_keys = list(test['visual_data_keys'])
	test_label = list(test['label'])

	train_keys = np.array(train_keys).astype(np.int64)
	print(train_keys.shape)
	test_keys = np.array(test_keys).astype(np.int64)
	print(test_keys.shape)

	train = pd.DataFrame()
	train['photo_id'] = train_keys
	train['label'] = train_label
	test = pd.DataFrame()
	test['photo_id'] = test_keys
	test['label'] = test_label

	return train,test

def oof_features(train_interaction,test_interaction,agg_col,target_col='click',use_mean=True,use_min=False,use_max=False,use_std=False,use_median=False,n_split=5,seed=1,split_col='user_id'):
    #计算oof特征，按照一列分组，分组后求均值
    skf =StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed).split(train_interaction[split_col],train_interaction[target_col])
    train_oof = np.zeros(train_interaction.shape[0])
    test_oof = np.zeros(test_interaction.shape[0])    
  
    for ind_tr, ind_te in skf:
        data_tr = train_interaction.iloc[ind_tr]
        data_te = train_interaction.iloc[ind_te]
        d = data_tr.groupby(agg_col)[target_col].mean().to_dict()
        train_oof[ind_te] = data_te[agg_col].apply(lambda x:d.get(x,0))
    
    d = train_interaction.groupby(agg_col)[target_col].mean().to_dict()
    test_oof = test_interaction[agg_col].apply(lambda x:d.get(x,0))
    
    return train_oof,test_oof

def oof_features2(train_interaction,test_interaction,agg_col,target_col='click',use_mean=True,use_min=False,use_max=False,use_std=False,use_median=False,n_split=5,seed=1,split_col='user_id'):
	#计算oof特征，按照两列分组，分组后求均值
    skf =StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed).split(train_interaction[split_col],train_interaction[target_col])
    train_oof = np.zeros(train_interaction.shape[0])
    test_oof = np.zeros(test_interaction.shape[0])  

    for ind_tr, ind_te in skf:
        data_tr = train_interaction.iloc[ind_tr]
        data_te = train_interaction.iloc[ind_te]
        d = data_tr.groupby(agg_col)[target_col].mean().to_dict()
        d_keys = set(d.keys())
        d2 = data_tr.groupby(agg_col[0])[target_col].mean().to_dict()
        tmp = []
        for i in data_te[agg_col].values:
            if tuple(i) in d_keys:
                tmp.append(d[tuple(i)])
            else:
                tmp.append(d2[i[0]])
        train_oof[ind_te]=tmp

    d = train_interaction.groupby(agg_col)[target_col].mean().to_dict()
    d_keys = set(d.keys())
    d2 = train_interaction.groupby(agg_col[0])[target_col].mean().to_dict()
    tmp = []
    for i in test_interaction[agg_col].values:
        if tuple(i) in d_keys:
            tmp.append(d[tuple(i)])
        else:
            tmp.append(d2[i[0]])
    test_oof = tmp

    return train_oof,test_oof

def oof_features_sum(train_interaction,test_interaction,agg_col,target_col='click',use_mean=True,use_min=False,use_max=False,use_std=False,use_median=False,n_split=5,seed=1,split_col='user_id'):
	#计算oof特征，按照一列分组，分组后求和
    skf =StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed).split(train_interaction[split_col],train_interaction[target_col])
    train_oof = np.zeros(train_interaction.shape[0])
    test_oof = np.zeros(test_interaction.shape[0])    
  
    for ind_tr, ind_te in skf:
        data_tr = train_interaction.iloc[ind_tr]
        data_te = train_interaction.iloc[ind_te]
        d = data_tr.groupby(agg_col)[target_col].sum().to_dict()
        train_oof[ind_te] = data_te[agg_col].apply(lambda x:d.get(x,0))
    
    d = train_interaction.groupby(agg_col)[target_col].sum().to_dict()
    test_oof = test_interaction[agg_col].apply(lambda x:d.get(x,0))
    
    return train_oof,test_oof

#特征列表
num_features = ['time','duration_time'] 
#读入交互特征
train_columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time']
train_interaction = pd.read_table('./train/train_interaction.txt', header=None)
train_interaction.columns = train_columns

test_columns = ['user_id', 'photo_id', 'time', 'duration_time']
test_interaction = pd.read_table('./test/test_interaction.txt', header=None)
test_interaction.columns = test_columns

#读入visual_label,和交互特征连接
train100,test100 = read_visual_label('./train/kmeans100_result_train0.pkl',
									'./train/kmeans100_result_train1.pkl',
									'./test/kmeans100_result_test.pkl')
train100.rename(columns={'label':'label100'},inplace = True)
test100.rename(columns={'label':'label100'},inplace = True)
train_interaction = pd.merge(train_interaction,train100,on='photo_id',how='left')
test_interaction = pd.merge(test_interaction,test100,on='photo_id',how='left')

train500,test500 = read_visual_label('./train/kmeans500_result_train0.pkl',
									'./train/kmeans500_result_train1.pkl',
									'./test/kmeans500_result_test.pkl')
train500.rename(columns={'label':'label500'},inplace = True)
test500.rename(columns={'label':'label500'},inplace = True)
train_interaction = pd.merge(train_interaction,train500,on='photo_id',how='left')
test_interaction = pd.merge(test_interaction,test500,on='photo_id',how='left')

#计算用户侧特征
#每个用户的平均点击率，oof
train_user_click,test_user_click = oof_features(train_interaction,test_interaction,agg_col='user_id',target_col='click',split_col='user_id')
corrscore = spearmanr(train_interaction['click'],train_user_click)
print('user_click_oof:',corrscore)
train_interaction['user_click_oof'] = train_user_click
test_interaction['user_click_oof'] = test_user_click
num_features+=['user_click_oof']
#每个用户的平均播放时长，oof
train_user_playing_time,test_user_playing_time = oof_features(train_interaction,test_interaction,agg_col='user_id',target_col='playing_time',split_col='user_id')
corrscore = spearmanr(train_interaction['click'],train_user_playing_time)
print('user_playing_time_oof',corrscore)
train_interaction['user_playing_time_oof'] = train_user_playing_time
test_interaction['user_playing_time_oof'] = test_user_playing_time
num_features+=['user_playing_time_oof']
#每个用户被推荐的视频总数
c = 'user_id'
d = train_interaction[c].value_counts().to_dict()
train_interaction['%s_count'%c] = train_interaction[c].apply(lambda x:d.get(x,0))
test_interaction['%s_count'%c] = test_interaction[c].apply(lambda x:d.get(x,0))
corrscore = spearmanr(train_interaction['click'],train_interaction['%s_count'%c])
print('%s_count'%c,corrscore)
num_features+=['%s_count'%c]
#每个用户被推荐的视频总数，oof
train_interaction['count'] = np.ones(train_interaction.shape[0])
test_interaction['count'] = np.ones(test_interaction.shape[0])
train_user_count,test_user_count = oof_features_sum(train_interaction,test_interaction,agg_col='user_id',target_col='count',split_col='user_id')
corrscore = spearmanr(train_interaction['click'],train_user_count)
print('user_count_oof:',corrscore)
train_interaction['user_count_oof'] = train_user_count
test_interaction['user_count_oof'] = test_user_count
num_features+=['user_count_oof']

#计算视频侧特征
#每个视频类别(100 or 500)的平均点击率、平均播放时长，oof
train_label100_click,test_label100_click = oof_features(train_interaction,test_interaction,agg_col='label100',target_col='click',split_col='user_id')
train_interaction['label100_click_oof'] = train_label100_click
test_interaction['label100_click_oof'] = test_label100_click
corrscore = spearmanr(train_interaction['click'],train_label100_click)
print('label100_click_oof:',corrscore)
num_features+=['label100_click_oof']

train_label100_playing_time,test_label100_playing_time = oof_features(train_interaction,test_interaction,agg_col='label100',target_col='playing_time',split_col='user_id')
train_interaction['label100_playing_time_oof'] = train_label100_playing_time
test_interaction['label100_playing_time_oof'] = test_label100_playing_time
corrscore = spearmanr(train_interaction['click'],train_label100_click)
print('label100_playing_time_oof:',corrscore)
num_features+=['label100_playing_time_oof']

train_label500_click,test_label500_click = oof_features(train_interaction,test_interaction,agg_col='label500',target_col='click',split_col='user_id')
train_interaction['label500_click_oof'] = train_label500_click
test_interaction['label500_click_oof'] = test_label500_click
corrscore = spearmanr(train_interaction['click'],train_label500_click)
print('label500_click_oof:',corrscore)
num_features+=['label500_click_oof']

train_label500_playing_time,test_label500_playing_time = oof_features(train_interaction,test_interaction,agg_col='label500',target_col='playing_time',split_col='user_id')
train_interaction['label500_playing_time_oof'] = train_label500_playing_time
test_interaction['label500_playing_time_oof'] = test_label500_playing_time
corrscore = spearmanr(train_interaction['click'],train_label500_click)
print('label500_playing_time_oof:',corrscore)
num_features+=['label500_playing_time_oof']

#计算用户和视频的组合特征
#每个用户在每个视频类别(100 or 500)的平均点击率、平均播放时长，oof
#缺失的用每个用户的平均点击率、平均播放时长填充
train_user_click,test_user_click = oof_features2(train_interaction,test_interaction,agg_col=['user_id','label100'],target_col='click',split_col='user_id')
train_interaction['user_label100_click_oof'] = train_user_click
test_interaction['user_label100_click_oof'] = test_user_click
corrscore = spearmanr(train_interaction['click'],train_user_click)
print('user_label100_click_oof:',corrscore)
num_features+=['user_label100_click_oof']

train_user_playing_time,test_user_playing_time = oof_features2(train_interaction,test_interaction,agg_col=['user_id','label100'],target_col='playing_time',split_col='user_id')
train_interaction['user_label100_playing_time_oof'] = train_user_playing_time
test_interaction['user_label100_playing_time_oof'] = test_user_playing_time
corrscore = spearmanr(train_interaction['click'],train_user_click)
print('user_label100_playing_time:',corrscore)
num_features+=['user_label100_playing_time']

train_user_click,test_user_click = oof_features2(train_interaction,test_interaction,agg_col=['user_id','label500'],target_col='click',split_col='user_id')
train_interaction['user_label500_click_oof'] = train_user_click
test_interaction['user_label500_click_oof'] = test_user_click
corrscore = spearmanr(train_interaction['click'],train_user_click)
print('user_label500_click_oof:',corrscore)
num_features+=['user_label500_click_oof']

train_user_playing_time,test_user_playing_time = oof_features2(train_interaction,test_interaction,agg_col=['user_id','label500'],target_col='playing_time',split_col='user_id')
train_interaction['user_label500_playing_time_oof'] = train_user_playing_time
test_interaction['user_label500_playing_time_oof'] = test_user_playing_time
corrscore = spearmanr(train_interaction['click'],train_user_click)
print('user_label500_playing_time:',corrscore)
num_features+=['user_label500_playing_time']

#一共14维特征
print(num_features)

#特征预处理
scaler = MinMaxScaler()
train_interaction[num_features] = scaler.fit_transform(train_interaction[num_features])
test_interaction[num_features] = scaler.transform(test_interaction[num_features])

#float64变为float32,节省空间
train_interaction[num_features] = train_interaction[num_features].astype(np.float32)
test_interaction[num_features] = test_interaction[num_features].astype(np.float32)

#保存
train_interaction.to_csv('./train/train_features.csv',header=True,sep='\t',index=False)
test_interaction.to_csv('./test/test_features.csv',header=True,sep='\t',index=False)