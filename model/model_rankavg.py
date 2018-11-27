#encoding:utf-8
from __future__ import division
from collections import defaultdict
from glob import glob
import pandas as pd
import sys

glob_files = './files/*'
result_path = 'result_yaoliang2.txt'
def kaggle_bag(glob_files, loc_outfile):
    with open(loc_outfile,"wb") as outfile:
        all_ranks = defaultdict(list)
        print(glob(glob_files))
        for i, glob_file in enumerate(glob(glob_files) ):
            file_ranks = []
            print("parsing:", glob_file)
            # sort glob_file by first column, ignoring the first line
            lines = open(glob_file).readlines()
            lines = sorted(lines)
            for e, line in enumerate(lines):
                r = line.strip().split(",")
                file_ranks.append((float(r[1]), e, r[0]) )
            for rank, item in enumerate(sorted(file_ranks) ):
                all_ranks[(item[1],item[2])].append(rank)
                
        average_ranks = []
        for k in sorted(all_ranks):
            #tmp_sum = 0.764237+0.764808+0.768464+0.768466
            #tmp = all_ranks[k][0]*0.764808/tmp_sum+all_ranks[k][1]*0.768464*tmp_sum+\
            #all_ranks[k][2]*0.768466/tmp_sum+all_ranks[k][3]*0.764237/tmp_sum
            average_ranks.append((sum(all_ranks[k])/len(all_ranks[k]),k))
        ranked_ranks = []
        for rank, k in enumerate(sorted(average_ranks)):
            ranked_ranks.append((k[1][0],k[1][1],rank/(len(average_ranks)-1)))
        for k in sorted(ranked_ranks):
            outfile.write("%s,%s\n"%(k[1],k[2]))
        print("wrote to %s"%loc_outfile)

def submission2result(sub_path,result_path):
    # sub_path:提交文件
    # result_path:排序均值集成的输入文件，在进行排序均值集成之前，需要将提交文件转换一下格式
    df = pd.read_table(sub_path,header=None,sep='\t')
    del df[0],df[1]
    df.to_csv(result_path,header=False,sep=',',index=True)

def result2submission(result_path,template_path,sub_path):
    #result_path:排序均值集成的输出文件
    #template_path:随便一个提交文件，主要取user_id和photo_id这两列
    #sub_path:排序均值集成后的提交文件
    result=pd.read_table(result_path,header=None,sep=',')
    df = pd.read_table(template_path,header=None,sep='\t')

    result.columns = ['id','score']
    result = result.sort_values('id')

    writefile = open(sub_path,'w')
    for line_df,line_re in zip(df.values,result['score'].values):
        temp = round(line_re,6)
        writefile.write("%s\t%s\t%s\n"%(str(int(line_df[0])),str(int(line_df[1])),str(temp)))

if __name__ == '__main__':
    submission2result(sub_path1,result_path1)
    kaggle_bag(glob_files, result_path)
    result2submission(result_path,template_path,sub_path2)