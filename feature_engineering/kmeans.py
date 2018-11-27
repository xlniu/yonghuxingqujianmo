#encoding:utf-8
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics import calinski_harabaz_score,silhouette_score

def kmeans_train():
	#训练
	train0 = pd.read_pickle('./train/train0_sample0.1.pkl')
	train1 = pd.read_pickle('./train/train1_sample0.1.pkl')
	test = pd.read_pickle('./test/test_sample0.1.pkl')

	data = []
	for vec in train0.values():
	    data.append(vec)
	for vec in train1.values():
	    data.append(vec)
	for vec in test.values():
	    data.append(vec)
	data = np.array(data)
	del train0,train1,test
	print("read data over")

	n_list = [100,500]
	for n in n_list:
		km = KMeans(n_clusters=n, init='k-means++', max_iter=100, n_init=10,verbose=1,n_jobs=16,precompute_distances=False)
		label = km.fit_predict(data)

		ch_score = calinski_harabaz_score(data, label)
		print("ch_score_%d:%f"%(n,ch_score))

		joblib.dump(km, "./data/km_%d.m"%n)

def kmeans_test(model_path,data_path,label_path):
	#测试
	km = joblib.load(model_path)
	visual_data = pickle.load(open(data_path,'r'))
	data = []
	visual_data_keys = visual_data.keys()
	for vec in visual_data.values():
	    data.append(vec)
	del visual_data
	label=km.predict(data)
	del data
	f = open(label_path,'w')
	pickle.dump({'visual_data_keys':visual_data_keys,'label':label},f)
	f.close()

if __name__ == '__main__':
	#训练
	kmeans_train()
	#测试
	model_path = './data/km_100.m'
	data_path = './train/train1.pkl'
	label_path = './train/kmeans100_result_train1.pkl'
	kmeans_test(model_path,data_path,label_path)