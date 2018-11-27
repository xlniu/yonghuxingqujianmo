#encoding:utf-8
import pickle
import random

def random_sample(data_path,save_path):
	visual_data = pickle.load(open(data_path,'r'))
	visual_data_sample = random.sample(visual_data.values(), len(visual_data)/10)
	f = open(save_path,'w')
	pickle.dump(visual_data_sample,f)
	f.close()

if __name__ == '__main__':
	data_path = './train/train1.pkl'
	save_path = './train/train1_sample0.1.pkl'
	random_sample(data_path,save_path)