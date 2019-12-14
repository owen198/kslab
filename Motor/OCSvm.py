
# coding: utf-8

# # New feature use in OCSvm



import glob
import os
import re
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os,re
datafile = []
subdata = []
# ?‡å?è¦å??ºæ??‰æ?æ¡ˆç??®é?
path = r'Z:\ç·šæ?1å» \DAQ308'
# ?–å??€?‰æ?æ¡ˆè?å­ç›®?„å?ç¨?subfolder = []
folder = listdir(path)
for items in folder:
    if ((items > '2018-09-01') & (items <= '2019-02-01')):
        subfolder.append(items)
# ä»¥è¿´?ˆè???for dirname in subfolder:
    # ?¢ç?æª”æ??„ç?å°è·¯å¾?    fullpath = join(path, dirname)
    fileC = [f for f in listdir(fullpath) if len(re.findall('.STD22' ,f))>0 and len(re.findall('.tdms$' ,f))>0 and len(re.findall('.rolling' ,f))>0 and len(re.findall('.æ»¾æŸ±' ,f))>0]
    for fileD in fileC:
        if len((fileD))>0:
            time = re.findall('.*[a-z]+-(.*).tdms$',fileD)[0]
            subdata.append([dirname,time, fileD])
#            print(dirname)
testalldata = pd.DataFrame(subdata, columns = ['date','time','file'])    
display(testalldata)



# # ?¿æ?NAN

# In[52]:


df = testrawdata.iloc[:,3:]
df1 = testrawdata[~np.isnan(df).any(axis=1)]


# In[53]:


# normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_raw_minmax = min_max_scaler.fit_transform(df1.iloc[:,3:])
N_raw_Data = pd.DataFrame(np.append(df1.iloc[:,:3],X_raw_minmax, axis = 1))
N_raw_Data.columns = ['Date','Time','Path']+feature_names+['doubletone_f','doubletone_v']
# N_raw_Data


# In[15]:


traindata = N_raw_Data[(N_raw_Data.Date > '2018-07-01') & (N_raw_Data.Date < '2018-12-30')]
neardata = N_raw_Data[(N_raw_Data.Date > '2018-1-1') & (N_raw_Data.Date < '2019-02-01')]
abnormaldata =N_raw_Data[(N_raw_Data.Date == '2019-02-01')]


# # Feature selection

# In[854]:

# # Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
mask = SelectKBest(chi2, k=10)
FS_feature = mask.fit_transform(X, y)
cols = mask.get_support()
all_features = feature_names+['doubletone_f','doubletone_v']
combine = pd.Series(all_features, index =cols)
FS_traindata = FS_feature[:traindata.shape[0]+1,:]
FS_neardata = FS_feature[traindata.shape[0]+1:traindata.shape[0]+neardata.shape[0]+1,:]
FS_abnormaldata = FS_feature[traindata.shape[0]+neardata.shape[0]+1:,:]
indexs = [i+3 for i, x in enumerate(cols) if x] 
for i in range(1,32):
    locals()['FS_testneardata_{:02d}'.format(i)] = locals()['testneardata_{:02d}'.format(i)].iloc[:,indexs]
print(FS_traindata.shape)
print(FS_neardata.shape)
print(FS_abnormaldata.shape)

indexs = [i for i, x in enumerate(cols) if x] 
print(indexs)
for i in indexs:
    print(all_features[i])
#     print(FS_feature[i])
#     print(traindata.iloc[0,i+3])
# In[987]:


feat_importances



def scorer_(Y_pred):
    a = (Y_pred[Y_pred == -1].size)/(Y_pred.size)
    return a*100

## ocsvm

# In[30]:


import pickle
from sklearn import svm
from sklearn.externals import joblib

clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01)
clf.fit(FS_traindata)

#joblib.dump(clf,'E:\\python\\Fomos\\Line 1\\22NTM\\20190201\\save_newfeature_22.pkl')
pre_FS_traindata = clf.predict(FS_traindata)
pre_FS_neardata = clf.predict(FS_neardata)
pre_FS_abnormaldata = clf.predict(FS_abnormaldata)

print('traindata: '+ str(scorer_(pre_FS_traindata)))
print('neardata: '+ str(scorer_(pre_FS_neardata)))
print('abnormaldata: '+ str(scorer_(pre_FS_abnormaldata)))

print('traindata: '+ str( FS_traindata.shape))
print('neardata: '+ str( FS_neardata.shape))
print('abnormaldata: '+ str( FS_abnormaldata.shape))

for i in range(1,32):
    if locals()['FS_testneardata_{:02d}'.format(i)].shape[0] > 0:
        df = locals()['FS_testneardata_{:02d}'.format(i)]
        locals()['pre_testneardata_{:02d}'.format(i)] = clf.predict(df)
        locals()['pre_score_{:02d}'.format(i)] = scorer_(locals()['pre_testneardata_{:02d}'.format(i)])
        locals()['shape_{:02d}'.format(i)] = df.shape
    else: 
        locals()['pred_testneardata_{:02d}'.format(i)] = -1
        locals()['pre_score_{:02d}'.format(i)] = -1
        locals()['shape_{:02d}'.format(i)] = 0
    
    print(str(i)+') '+str(locals()['pre_score_{:02d}'.format(i)]) + ', shape: '+ str(locals()['shape_{:02d}'.format(i)]))


