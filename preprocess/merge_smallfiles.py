import os
import gc
import math
import pickle
import numpy as np

def readtest_savedata(filepath,savename):
    data = {}
    pathdir = os.listdir(filepath+'final_visual_test/')
    for eachdir in pathdir:
        data[eachdir] = np.load(os.path.join('%s%s') % (filepath, eachdir))[0]
    pickle.dump(data,open(os.path.join('%s%s')%(filepath,savename),"w"))

def readtrain_savedata(filepath,savename,splitnum=2):
    pathdir = os.listdir(filepath+'final_visual_train/')
    number = int(math.ceil(len(pathdir)*1.0/(splitnum*1.0)))
    for i in range(splitnum):
        data = {}
        for eachdir in pathdir[i*number:min((i+1)*number,len(pathdir))]:
            data[eachdir] = np.load(os.path.join('%sfinal_visual_train/%s') % (filepath, eachdir))[0]
        pickle.dump(data,open(os.path.join('%s%s%s.pkl')%(filepath,savename,str(i)),"w"))
        del data
        gc.collect()

if __name__ == '__main__':

    readtrain_savedata('./train/', 'train', 2)
    readtest_savedata('./test/', 'test.pkl')