# 用户兴趣建模大赛 top10 开源代码
## preprocess
### merge_smallfiles.py
   功能：将视觉特征的小文件合并
   输入：视觉特征，存放于单独的小文件中
   输出：train0.pkl，train1.pkl，test.pkl
   说明：由于内存限制，将训练集的视觉特征存放在两个文件中

### random_sample.py
   功能：分别在训练集和测试集视觉特征中采样10%，用于视觉特征聚类
   输入：train0.pkl，train1.pkl，test.pkl
   输出：train0_sample0.1.pkl，train1_sample0.1.pkl，test_sample0.1.pkl

## feature_engineering
### kmeans.py
   功能：在10%的视觉特征上训练kmeans聚类算法，并对所有视觉特征进行聚类，聚类个数分别为100和500
   输入：train0_sample0.1.pkl，train1_sample0.1.pkl，test_sample0.1.pkl，
	     train0.pkl，train1.pkl，test.pkl
   输出：kmeans100_result_train0.pkl，kmeans100_result_train1.pkl，kmeans100_result_test.pkl
         kmeans500_result_train0.pkl，kmeans500_result_train1.pkl，kmeans500_result_test.pkl
### interaction_feature.py
   功能：计算统计特征(共14维特征)
   输入：train_interaction.txt，test_interaction.txt
	     kmeans100_result_train0.pkl，kmeans100_result_train1.pkl，kmeans100_result_test.pkl
	     kmeans500_result_train0.pkl，kmeans500_result_train1.pkl，kmeans500_result_test.pkl
   输出：train_features.csv，test_features.csv
### user_photo_emb.py
   功能：使用deepwalk算法，计算user.emb和photo.emb
   输入：train_interaction.txt，test_interaction.txt
   输出：user.emb，photo.emb
         user_doc.csv，photo_doc.csv
   说明：user.emb和photo.emb是根据用户的交互信息计算出来的64维embedding特征
         user_doc.csv是每个视频推荐给的用户列表
         photo_doc.csv是每个用户被推荐的视频列表(按照时间戳排序)
### text_feature.py
   功能：提取停用词，计算文字描述的tfidf特征，再使用nmf算法提取主题分布，主题个数为20个
   输入：train_interaction.txt，test_interaction.txt
	     train_text.txt，test_text.txt
   输出：text_nmf20.pkl
### face_feature.py
   功能：提取人脸特征
   输入：train_interaction.txt，test_interaction.txt
         train_face.txt，test_face.txt
   输出：face35.pkl

## model
### model1.py
   功能：模型1，按照时间序列将原始训练集分为训练集和验证集，训练集数据占原始训练集的80%，使用验证集的auc进行提前停止
   输入：train_interaction.txt，test_interaction.txt
	     user_doc.csv
	     user.emb，photo.emb
   输出：submission1.txt
### model2.py
   功能：模型2，按照时间序列将原始训练集分为训练集和验证集，训练集数据占原始训练集的80%，使用验证集的auc进行提前停止
   输入：train_interaction.txt，test_interaction.txt
	     user_doc.csv，photo_doc.csv
	     user.emb，photo.emb
   输出：submission2.txt
### model3.py
   功能：模型3，按照时间序列将原始训练集分为训练集和验证集，训练集数据占原始训练集的80%，使用验证集的auc进行提前停止
   输入：train_interaction.txt，test_interaction.txt
	     user_doc.csv，photo_doc.csv
	     user.emb，photo.emb
	     text_nmf20.pkl
   输出：submission3.txt
### model4.py
   功能：模型4，按照时间序列将原始训练集分为训练集和验证集，训练集数据占原始训练集的80%，使用验证集的auc进行提前停止
   输入：train_interaction.txt，test_interaction.txt
	     user_doc.csv，photo_doc.csv
	     user.emb，photo.emb
	     face35.pkl
   输出：submission4.txt
### model5.py
   功能：模型5，使用训练集的全量数据训练模型，使用测试集和线上最好结果的spearman系数进行提前停止
   输入：train_interaction.txt，test_interaction.txt
	     train_features.csv，test_features.csv
	     user_doc.csv
	     user.emb，photo.emb
   输出：submission5.txt
   说明：线上最好结果为model1-model4的排序均值集成
### model6.py
   功能：模型6，使用训练集的全量数据训练模型，使用测试集和线上最好结果的spearman系数进行提前停止
   输入：train_interaction.txt，test_interaction.txt
	     train_features.csv，test_features.csv
	     user_doc.csv，photo_doc.csv
	     user.emb，photo.emb
   输出：submission6.txt
### model7.py
   功能：模型7，使用训练集的全量数据训练模型，使用测试集和线上最好结果的spearman系数进行提前停止
   输入：train_interaction.txt，test_interaction.txt
	     train_features.csv，test_features.csv
	     user_doc.csv，photo_doc.csv
	     user.emb，photo.emb
	     text_nmf20.pkl
   输出：submission7.txt
### model8.py
   功能：模型8，使用训练集的全量数据训练模型，使用测试集和线上最好结果的spearman系数进行提前停止
   输入：train_interaction.txt，test_interaction.txt
	     train_features.csv，test_features.csv
	     user_doc.csv，photo_doc.csv
	     user.emb，photo.emb
	     face35.pkl
   输出：submission8.txt
### model_rankavg.py
   功能：排序均值集成
   输入：包含所有单模型结果的文件夹名称
   输出：集成结果文件
