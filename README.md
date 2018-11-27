# 用户兴趣建模大赛 top10 开源代码
代码分为三个部分：数据预处理、特征工程、模型。下面分别列出了这三部分对应的代码文件，并对每个代码文件的功能、输入、输出做了详细说明。
## preprocess
### merge_smallfiles.py
   功能：将视觉特征的小文件合并;<br>
   输入：视觉特征，存放于单独的小文件中;<br>
   输出：train0.pkl，train1.pkl，test.pkl<br>
   说明：由于内存限制，将训练集的视觉特征存放在两个文件中;<br>
### random_sample.py
   功能：分别在训练集和测试集视觉特征中采样10%，用于视觉特征聚类;<br>
   输入：train0.pkl，train1.pkl，test.pkl<br>
   输出：train0_sample0.1.pkl，train1_sample0.1.pkl，test_sample0.1.pkl<br>
## feature_engineering
### kmeans.py
   功能：在10%的视觉特征上训练kmeans聚类算法，并对所有视觉特征进行聚类，聚类个数分别为100和500;<br>
   输入：train0_sample0.1.pkl，train1_sample0.1.pkl，test_sample0.1.pkl，<br>
	 &emsp;&emsp;&emsp;train0.pkl，train1.pkl，test.pkl<br>
   输出：kmeans100_result_train0.pkl，kmeans100_result_train1.pkl，kmeans100_result_test.pkl<br>
        &emsp;&emsp;&emsp;kmeans500_result_train0.pkl，kmeans500_result_train1.pkl，kmeans500_result_test.pkl<br>
### interaction_feature.py
   功能：计算统计特征(共14维特征);<br>
   输入：train_interaction.txt，test_interaction.txt<br>
	 &emsp;&emsp;&emsp;kmeans100_result_train0.pkl，kmeans100_result_train1.pkl，kmeans100_result_test.pkl<br>
	 &emsp;&emsp;&emsp;kmeans500_result_train0.pkl，kmeans500_result_train1.pkl，kmeans500_result_test.pkl<br>
   输出：train_features.csv，test_features.csv<br>
### user_photo_emb.py
   功能：使用deepwalk算法，计算user.emb和photo.emb;<br>
   输入：train_interaction.txt，test_interaction.txt<br>
   输出：user.emb，photo.emb<br>
        &emsp;&emsp;&emsp;user_doc.csv，photo_doc.csv<br>
   说明：user.emb和photo.emb是根据用户的交互信息计算出来的64维embedding特征;<br>
        &emsp;&emsp;&emsp;user_doc.csv是每个视频推荐给的用户列表;<br>
        &emsp;&emsp;&emsp;photo_doc.csv是每个用户被推荐的视频列表(按照时间戳排序);<br>
### text_feature.py
   功能：提取停用词，计算文字描述的tfidf特征，再使用nmf算法提取主题分布，主题个数为20个;<br>
   输入：train_interaction.txt，test_interaction.txt<br>
	&emsp;&emsp;&emsp;train_text.txt，test_text.txt<br>
   输出：text_nmf20.pkl<br>
### face_feature.py
   功能：提取人脸特征;<br>
   输入：train_interaction.txt，test_interaction.txt<br>
         &emsp;&emsp;&emsp;train_face.txt，test_face.txt<br>
   输出：face35.pkl<br>
## model
### model1.py
   功能：模型1，按照时间序列将原始训练集分为训练集和验证集，训练集数据占原始训练集的80%，使用验证集的auc进行提前停止;<br>
   输入：train_interaction.txt，test_interaction.txt<br>
	 &emsp;&emsp;&emsp;user_doc.csv<br>
	 &emsp;&emsp;&emsp;user.emb，photo.emb<br>
   输出：submission1.txt<br>
### model2.py
   功能：模型2，按照时间序列将原始训练集分为训练集和验证集，训练集数据占原始训练集的80%，使用验证集的auc进行提前停止;<br>
   输入：train_interaction.txt，test_interaction.txt<br>
	 &emsp;&emsp;&emsp;user_doc.csv，photo_doc.csv<br>
	 &emsp;&emsp;&emsp;user.emb，photo.emb<br>
   输出：submission2.txt<br>
### model3.py
   功能：模型3，按照时间序列将原始训练集分为训练集和验证集，训练集数据占原始训练集的80%，使用验证集的auc进行提前停止;<br>
   输入：train_interaction.txt，test_interaction.txt<br>
	 &emsp;&emsp;&emsp;user_doc.csv，photo_doc.csv<br>
	 &emsp;&emsp;&emsp;user.emb，photo.emb<br>
	 &emsp;&emsp;&emsp;text_nmf20.pkl<br>
   输出：submission3.txt<br>
### model4.py
   功能：模型4，按照时间序列将原始训练集分为训练集和验证集，训练集数据占原始训练集的80%，使用验证集的auc进行提前停止;<br>
   输入：train_interaction.txt，test_interaction.txt<br>
	 &emsp;&emsp;&emsp;user_doc.csv，photo_doc.csv<br>
	 &emsp;&emsp;&emsp;user.emb，photo.emb<br>
	 &emsp;&emsp;&emsp;face35.pkl<br>
   输出：submission4.txt<br>
### model5.py
   功能：模型5，使用训练集的全量数据训练模型，使用测试集和线上最好结果的spearman系数进行提前停止;<br>
   输入：train_interaction.txt，test_interaction.txt<br>
	 &emsp;&emsp;&emsp;train_features.csv，test_features.csv<br>
	 &emsp;&emsp;&emsp;user_doc.csv<br>
	 &emsp;&emsp;&emsp;user.emb，photo.emb<br>
   输出：submission5.txt<br>
   说明：线上最好结果为model1-model4的排序均值集成;<br>
### model6.py
   功能：模型6，使用训练集的全量数据训练模型，使用测试集和线上最好结果的spearman系数进行提前停止;<br>
   输入：train_interaction.txt，test_interaction.txt<br>
	     &emsp;&emsp;&emsp;train_features.csv，test_features.csv<br>
	     &emsp;&emsp;&emsp;user_doc.csv，photo_doc.csv<br>
	     &emsp;&emsp;&emsp;user.emb，photo.emb<br>
   输出：submission6.txt<br>
### model7.py
   功能：模型7，使用训练集的全量数据训练模型，使用测试集和线上最好结果的spearman系数进行提前停止;<br>
   输入：train_interaction.txt，test_interaction.txt<br>
	     &emsp;&emsp;&emsp;train_features.csv，test_features.csv<br>
	     &emsp;&emsp;&emsp;user_doc.csv，photo_doc.csv<br>
	     &emsp;&emsp;&emsp;user.emb，photo.emb<br>
	     &emsp;&emsp;&emsp;text_nmf20.pkl<br>
   输出：submission7.txt<br>
### model8.py
   功能：模型8，使用训练集的全量数据训练模型，使用测试集和线上最好结果的spearman系数进行提前停止;<br>
   输入：train_interaction.txt，test_interaction.txt<br>
	     &emsp;&emsp;&emsp;train_features.csv，test_features.csv<br>
	     &emsp;&emsp;&emsp;user_doc.csv，photo_doc.csv<br>
	     &emsp;&emsp;&emsp;user.emb，photo.emb<br>
	     &emsp;&emsp;&emsp;face35.pkl<br>
   输出：submission8.txt<br>
### model_rankavg.py
   功能：排序均值集成;<br>
   输入：包含所有单模型结果的文件夹名称;<br>
   输出：集成结果文件;<br>
